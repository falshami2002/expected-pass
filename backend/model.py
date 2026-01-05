from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import register_keras_serializable


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


def build_pass_model(n_players: int, d_player: int, d_global: int) -> tf.keras.Model:
    players_in = layers.Input(shape=(n_players, d_player), name="players")
    mask_in = layers.Input(shape=(n_players,), name="players_mask")
    global_in = layers.Input(shape=(d_global,), name="global_feats")

    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(players_in)
    x = layers.TimeDistributed(layers.Dense(64, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)

    mask_exp = layers.Lambda(expand_last, name="expand_last_mask")(mask_in)
    x_masked = layers.Multiply(name="apply_mask")([x, mask_exp])

    denom = layers.Lambda(mask_denom, name="mask_denom")(mask_exp)
    mean_pool = layers.Lambda(mean_pool_fn, name="mean_pool")([x_masked, denom])
    max_pool = layers.Lambda(max_pool_fn, name="max_pool")(x_masked)

    att_scores = layers.TimeDistributed(layers.Dense(1), name="att_scores")(x)
    att_scores = layers.Lambda(squeeze_last, name="squeeze_scores")(att_scores)
    att_weights = layers.Lambda(masked_softmax_layer, name="masked_softmax")(
        [att_scores, mask_in]
    )
    att_weightsE = layers.Lambda(expand_last, name="expand_last_att")(att_weights)
    att_pool = layers.Lambda(weighted_sum, name="att_weighted_sum")(
        [x, att_weightsE]
    )

    set_repr = layers.Concatenate(name="set_repr")(
        [mean_pool, max_pool, att_pool]
    )
    h = layers.Concatenate(name="concat_globals")([set_repr, global_in])

    h = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(h)
    h = layers.Dropout(0.25)(h)
    h = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
    )(h)
    h = layers.Dropout(0.25)(h)
    out = layers.Dense(1, activation="sigmoid")(h)

    model = models.Model(
        inputs=[players_in, mask_in, global_in],
        outputs=out,
        name="xpass_model",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    return model


def load_xpass_model(
    weights_path: str | Path,
    *,
    n_players: int = 22,
    d_player: int = 9,
    d_global: int = 16,
) -> tf.keras.Model:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    model = build_pass_model(
        n_players=n_players,
        d_player=d_player,
        d_global=d_global,
    )
    model.load_weights(str(weights_path))
    return model
