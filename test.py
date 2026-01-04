import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import train_test_split

# ---------- Named, serializable tensor ops (no inline lambdas) ----------
@register_keras_serializable(package="xpass")
def masked_softmax(logits, mask, axis=-1):
    neg_inf = tf.fill(tf.shape(logits), tf.constant(-1e9, dtype=logits.dtype))
    masked_logits = tf.where(tf.cast(mask, tf.bool), logits, neg_inf)
    return tf.nn.softmax(masked_logits, axis=axis)

@register_keras_serializable(package="xpass")
def expand_last(x):
    return tf.expand_dims(x, -1)

@register_keras_serializable(package="xpass")
def squeeze_last(x):
    return tf.squeeze(x, -1)

@register_keras_serializable(package="xpass")
def mask_denom(m):
    # sum mask over players, keep dims for broadcasting, clip to avoid /0
    s = tf.reduce_sum(m, axis=1, keepdims=True)
    return tf.clip_by_value(s, 1e-6, 1e9)

@register_keras_serializable(package="xpass")
def mean_pool_fn(inputs):
    x_masked, denom = inputs
    return tf.reduce_sum(x_masked, axis=1) / tf.squeeze(denom, axis=1)

@register_keras_serializable(package="xpass")
def max_pool_fn(x):
    return tf.reduce_max(x, axis=1)

@register_keras_serializable(package="xpass")
def masked_softmax_layer(inputs):
    scores, mask = inputs
    return masked_softmax(scores, mask, axis=1)

@register_keras_serializable(package="xpass")
def weighted_sum(inputs):
    x, w = inputs
    return tf.reduce_sum(x * w, axis=1)

# ---------- Model ----------
def build_pass_model(n_players: int, d_player: int, d_global: int) -> tf.keras.Model:
    players_in = layers.Input(shape=(n_players, d_player), name="players")
    mask_in    = layers.Input(shape=(n_players,),       name="players_mask")
    global_in  = layers.Input(shape=(d_global,),        name="global_feats")

    # Per-player MLP (shared)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(players_in)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)  # (B, N, 32)

    # Apply mask
    mask_exp = layers.Lambda(expand_last, name="expand_last_mask")(mask_in)  # (B,N,1)
    x_masked = layers.Multiply(name="apply_mask")([x, mask_exp])

    # DeepSets pooling
    denom    = layers.Lambda(mask_denom, name="mask_denom")(mask_exp)        # (B,1,1)
    mean_pool= layers.Lambda(mean_pool_fn, name="mean_pool")([x_masked, denom])  # (B,32)
    max_pool = layers.Lambda(max_pool_fn,  name="max_pool")(x_masked)            # (B,32)

    # Attention pooling (masked)
    att_scores   = layers.TimeDistributed(layers.Dense(1), name="att_scores")(x)  # (B,N,1)
    att_scores   = layers.Lambda(squeeze_last, name="squeeze_scores")(att_scores) # (B,N)
    att_weights  = layers.Lambda(masked_softmax_layer, name="masked_softmax")([att_scores, mask_in])  # (B,N)
    att_weightsE = layers.Lambda(expand_last, name="expand_last_att")(att_weights)                    # (B,N,1)
    att_pool     = layers.Lambda(weighted_sum, name="att_weighted_sum")([x, att_weightsE])            # (B,32)

    # Concatenate set representations + globals
    set_repr = layers.Concatenate(name="set_repr")([mean_pool, max_pool, att_pool])  # (B,96)
    h = layers.Concatenate(name="concat_globals")([set_repr, global_in])

    # Head
    h = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(h)
    h = layers.Dropout(0.25)(h)
    out = layers.Dense(1, activation="sigmoid")(h)

    model = models.Model(inputs=[players_in, mask_in, global_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ---------- Train & Save ----------
bundle = np.load("xpass_nn_inputs.npz", allow_pickle=True)
players_np = bundle["players"]   # (N, 22, 9)
mask_np    = bundle["mask"]      # (N, 22)
globals_np = bundle["globals"]   # (N, 16)
y_np       = bundle["y"]         # (N,)

P_train, P_val, M_train, M_val, G_train, G_val, y_train, y_val = train_test_split(
    players_np, mask_np, globals_np, y_np, test_size=0.2, random_state=42, stratify=y_np
)

# ... your imports + build_pass_model + train/val split ...

model = build_pass_model(n_players=22, d_player=players_np.shape[-1], d_global=globals_np.shape[-1])

# Save ONLY weights (recommended)
ckpt_weights = tf.keras.callbacks.ModelCheckpoint(
    "xpass_best_weights.weights.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)

# (Optional) also save the full model as H5
ckpt_full = tf.keras.callbacks.ModelCheckpoint(
    "xpass_best_full.h5",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,   # full model
    verbose=1
)

early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)

history = model.fit(
    {"players": P_train, "players_mask": M_train, "global_feats": G_train}, y_train,
    validation_data=({"players": P_val, "players_mask": M_val, "global_feats": G_val}, y_val),
    epochs=200, batch_size=64,
    callbacks=[early, ckpt_weights, ckpt_full],   # use one or both
    verbose=1
)

# Final saves (optional)
model.save_weights("xpass_final_weights.weights.h5")
model.save("xpass_final_full.h5")   # full model H5
