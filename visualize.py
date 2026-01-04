import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from kloppy import metrica
from test3 import build_pass_model

# Your existing helpers
from get_features import (
    get_xy,
    nearest_def_to_point,
    defenders_in_radius_point,
    min_lane_block,
    cone_pressure_in_pass_dir,
)

FRAME_WIN = 150  # ~6s at 25 fps

def infer_pass_success_from_events(events_df: pd.DataFrame, pass_row: pd.Series, frame_win: int = FRAME_WIN) -> bool:
    """
    Returns True if the pass was successful in reality, else False.
    Uses the same windowed look-ahead logic we used earlier.
    """
    # Ensure normalized columns exist
    if "event_type_norm" not in events_df:
        events_df = events_df.copy()
        events_df["event_type_norm"] = events_df["event_type"].astype(str).str.lower().str.strip()
        events_df["subtype_norm"] = events_df["subtype"].astype(str).str.lower().str.strip()

    pid = int(pass_row["period_id"])
    sf  = int(pass_row["start_frame"])
    ef  = int(pass_row["end_frame"])
    team_raw = str(pass_row["team_id"]).strip().lower()

    g = events_df[events_df["period_id"] == pid].copy()
    g = g.sort_values("start_frame")

    # window after the pass end frame
    in_win = (g["start_frame"] > ef) & (g["start_frame"] <= ef + frame_win)
    if not in_win.any():
        return True  # default to success if nothing else happens in the window

    et = g.loc[in_win, "event_type_norm"].to_numpy()
    st = g.loc[in_win, "subtype_norm"].to_numpy()
    tm = g.loc[in_win, "team_id"].astype(str).str.lower().to_numpy()

    # explicit fail events in the window?
    is_fail_evt = (et == "ball out") | ((et == "ball lost") & np.isin(st, ["interception", "head interception"]))
    if is_fail_evt.any():
        return False

    # first on-ball event; if by opponent → fail, else success
    ONBALL = {"pass", "shot", "dribble", "carry", "recovery"}
    is_onball = np.isin(et, list(ONBALL))

    idx = np.where(is_onball)[0]
    if idx.size:
        j = idx[0]
        return bool(tm[j] == team_raw)

    # nothing decisive → assume success
    return True


# ---------------- Pitch drawing ----------------
def draw_pitch(ax, xlim=(0,1), ylim=(0,1)):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    # Basic lines (Metrica normalized [0,1])
    ax.plot([0,1],[0,0], 'k-', lw=1)
    ax.plot([0,1],[1,1], 'k-', lw=1)
    ax.plot([0,0],[0,1], 'k-', lw=1)
    ax.plot([1,1],[0,1], 'k-', lw=1)
    ax.plot([0.5,0.5],[0,1], 'k--', lw=0.7)
    # Center circle (approx)
    circle = plt.Circle((0.5,0.5), 0.0915, color='k', fill=False, lw=0.7)  # ~9.15m in normalized coords if field is 105x68 → scaled ~ 0.0915 horizontally
    ax.add_patch(circle)
    return ax

# ---------------- Roster helpers ----------------
def get_all_team_players(tracking):
    home, away = tracking.metadata.teams[0], tracking.metadata.teams[1]
    home_ids = [p.player_id for p in home.players]
    away_ids = [p.player_id for p in away.players]
    return home_ids, away_ids, home, away

def _fix11(ids):
    ids = list(ids)
    if len(ids) >= 11: return ids[:11]
    return ids + [f"PAD_{i}" for i in range(11 - len(ids))]

def build_player_tensor_for_tr_row(
    tr_row, passer_pid, receiver_pid, home_ids, away_ids, px, py, bx, by
):
    # order: passer's team first
    if passer_pid in home_ids:
        home_fixed = _fix11(home_ids); away_fixed = _fix11(away_ids)
        order = home_fixed + away_fixed
        team_flag = {pid: 1 for pid in home_fixed} | {pid: 0 for pid in away_fixed}
    elif passer_pid in away_ids:
        away_fixed = _fix11(away_ids); home_fixed = _fix11(home_ids)
        order = away_fixed + home_fixed
        team_flag = {pid: 1 for pid in away_fixed} | {pid: 0 for pid in home_fixed}
    else:
        home_fixed = _fix11(home_ids); away_fixed = _fix11(away_ids)
        order = home_fixed + away_fixed
        team_flag = {pid: 1 if pid in home_fixed else 0 for pid in order}

    d_player = 9
    feats = np.zeros((22, d_player), dtype=np.float32)
    mask  = np.zeros((22,), dtype=np.float32)
    bx = float(bx) if bx is not None and not pd.isna(bx) else np.nan
    by = float(by) if by is not None and not pd.isna(by) else np.nan

    for i, pid in enumerate(order):
        if isinstance(pid, str) and pid.startswith("PAD_"):
            continue

        x, y = get_xy(tr_row, pid)  # expects metrica-normalized coords
        if pd.isna(x) or pd.isna(y):
            continue

        dx_p = 0.0 if (pd.isna(px) or pd.isna(py)) else float(x) - float(px)
        dy_p = 0.0 if (pd.isna(px) or pd.isna(py)) else float(y) - float(py)

        dx_b = 0.0 if (pd.isna(bx) or pd.isna(by)) else float(x) - float(bx)
        dy_b = 0.0 if (pd.isna(bx) or pd.isna(by)) else float(y) - float(by)

        feats[i, 0] = float(x)
        feats[i, 1] = float(y)
        feats[i, 2] = dx_p
        feats[i, 3] = dy_p
        feats[i, 4] = dx_b
        feats[i, 5] = dy_b
        feats[i, 6] = float(team_flag.get(pid, 0))
        feats[i, 7] = 1.0 if pid == passer_pid else 0.0
        feats[i, 8] = 1.0 if (pd.notna(receiver_pid) and pid == receiver_pid) else 0.0
        mask[i] = 1.0

    return feats, mask

def build_globals_for_tr_row(
    tr_row, passer_pid, receiver_pid, opp_ids
):
    # passer
    px, py = get_xy(tr_row, passer_pid)
    # receiver: if missing at this exact frame, skip by returning None
    rx, ry = (np.nan, np.nan)
    if pd.notna(receiver_pid):
        rx, ry = get_xy(tr_row, receiver_pid)
    if pd.isna(px) or pd.isna(py) or pd.isna(rx) or pd.isna(ry):
        return None  # can't build globals for this frame cleanly

    # ball
    bx, by = tr_row.get("ball_x"), tr_row.get("ball_y")

    # geometry
    pass_distance = float(np.hypot(float(rx) - float(px), float(ry) - float(py)))
    pass_angle = float(np.arctan2(float(ry) - float(py), float(rx) - float(px)))
    pass_angle_sin, pass_angle_cos = np.sin(pass_angle), np.cos(pass_angle)

    # defensive / lane features (your functions)
    rec_near_def = nearest_def_to_point((rx, ry), opp_ids, tr_row)
    rec_def_5    = defenders_in_radius_point((rx, ry), opp_ids, tr_row, r=0.05)
    lane_min     = min_lane_block((px, py), (rx, ry), opp_ids, tr_row)
    passer_near_def = nearest_def_to_point((px, py), opp_ids, tr_row)
    passer_def_r4   = defenders_in_radius_point((px, py), opp_ids, tr_row, r=0.04)
    cone_cnt, cone_min_lat = cone_pressure_in_pass_dir(
        (px, py), (rx, ry), opp_ids, tr_row, max_dist=0.20, half_angle_deg=30
    )

    g = np.array([
        float(px), float(py), float(rx), float(ry),
        float(bx) if bx is not None and not pd.isna(bx) else 0.0,
        float(by) if by is not None and not pd.isna(by) else 0.0,
        pass_distance, pass_angle_sin, pass_angle_cos,
        float(rec_near_def) if np.isfinite(rec_near_def) else 0.0,
        float(rec_def_5) if np.isfinite(rec_def_5) else 0.0,
        float(lane_min) if np.isfinite(lane_min) else 0.0,
        float(passer_near_def) if np.isfinite(passer_near_def) else 0.0,
        float(passer_def_r4) if np.isfinite(passer_def_r4) else 0.0,
        float(cone_cnt) if np.isfinite(cone_cnt) else 0.0,
        float(cone_min_lat) if np.isfinite(cone_min_lat) else 0.0,
    ], dtype=np.float32)
    return g

# ---------------- Windowing & animation ----------------
def get_tracking_window(df_tracking, period_id, center_frame, before=20, after=20):
    g = df_tracking[df_tracking["period_id"] == period_id].copy()
    g = g.sort_values("frame_id")
    start_f = center_frame - before
    end_f   = center_frame + after
    return g[(g["frame_id"] >= start_f) & (g["frame_id"] <= end_f)].copy()

from matplotlib.patches import FancyArrowPatch

def animate_pass_sequence(
    match_id: int,
    period_id: int,
    center_frame: int,
    passer_id: str,
    receiver_id: str,
    model,
    save_path="pass_anim.gif",
    frames_before=20,
    frames_after=20,
    dynamic=True,
    speed_factor=0.25,   # 0.25x speed (4x slower)
    actual_success: bool = True
):
    # Load tracking & meta
    tracking = metrica.load_open_data(match_id=match_id, coordinates="metrica")
    df_tr = tracking.to_df().sort_values(["period_id", "frame_id"])
    home_ids, away_ids, home, away = get_all_team_players(tracking)

    # Opponent set (by passer team)
    passer_on_home = (passer_id in home_ids)
    opp_ids = away_ids if passer_on_home else home_ids

    # Extract window
    win = get_tracking_window(df_tr, period_id, center_frame, frames_before, frames_after)
    if win.empty:
        raise ValueError("No tracking rows found for the requested window. Check frame_id/period_id.")

    # Precompute per-frame positions + predictions
    xs_home, ys_home, xs_away, ys_away = [], [], [], []
    ball_xs, ball_ys = [], []
    xpass_per_frame = []

    for _, row in win.iterrows():
        # team dots
        hx, hy, axx, ayy = [], [], [], []
        for pid in home_ids:
            x, y = get_xy(row, pid)
            if not (pd.isna(x) or pd.isna(y)): hx.append(float(x)); hy.append(float(y))
        for pid in away_ids:
            x, y = get_xy(row, pid)
            if not (pd.isna(x) or pd.isna(y)): axx.append(float(x)); ayy.append(float(y))
        xs_home.append(hx); ys_home.append(hy)
        xs_away.append(axx); ys_away.append(ayy)

        # ball
        bx, by = row.get("ball_x"), row.get("ball_y")
        ball_xs.append(float(bx) if bx is not None and not pd.isna(bx) else np.nan)
        ball_ys.append(float(by) if by is not None and not pd.isna(by) else np.nan)

        # prediction
        if dynamic or (row["frame_id"] == center_frame):
            px, py = get_xy(row, passer_id)
            feats, mask = build_player_tensor_for_tr_row(row, passer_id, receiver_id, home_ids, away_ids, px, py, bx, by)
            gvec = build_globals_for_tr_row(row, passer_id, receiver_id, opp_ids)
            if gvec is None:
                xpass = np.nan
            else:
                xpass = float(model.predict(
                    {"players": feats[None, ...],
                     "players_mask": mask[None, ...],
                     "global_feats": gvec[None, ...]},
                    verbose=0
                ).ravel()[0])
        else:
            xpass = np.nan
        xpass_per_frame.append(xpass)

    if not dynamic:
        # freeze prediction at pass frame
        center_idx = int(np.argmin(np.abs(win["frame_id"].values - center_frame)))
        center_val = xpass_per_frame[center_idx]
        xpass_per_frame = [center_val for _ in xpass_per_frame]

    # ---------- Matplotlib animation ----------
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_pitch(ax)

    home_sc = ax.scatter([], [], s=35, marker='o', edgecolor='k', linewidths=0.5, label='Home')
    away_sc = ax.scatter([], [], s=35, marker='s', edgecolor='k', linewidths=0.5, label='Away')
    ball_sc = ax.scatter([], [], s=20, marker='*', edgecolor='k', linewidths=0.5, label='Ball')

    # Moving pass line (updates every frame). Using a simple Line2D (clean + fast).
    pass_line, = ax.plot([], [], lw=2, linestyle='-', color='tab:blue')

    title = ax.text(0.02, 1.02, "", transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
    xpass_text = ax.text(0.98, 1.02, "", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, fontweight="bold")
    actual_text = ax.text(0.98, 0.97,
                          "Actual: SUCCESS" if actual_success else "Actual: FAIL",
                          transform=ax.transAxes, ha="right", va="top",
                          fontsize=11, fontweight="bold",
                          color=("tab:green" if actual_success else "tab:red"))

    frames_vals = win["frame_id"].values

    def init():
        home_sc.set_offsets(np.c_[[], []])
        away_sc.set_offsets(np.c_[[], []])
        ball_sc.set_offsets(np.c_[[], []])
        pass_line.set_data([], [])
        title.set_text("")
        xpass_text.set_text("")
        return home_sc, away_sc, ball_sc, pass_line, title, xpass_text, actual_text

    def update(i):
        row = win.iloc[i]

        # scatter updates
        if xs_home[i]:
            home_sc.set_offsets(np.c_[xs_home[i], ys_home[i]])
        else:
            home_sc.set_offsets(np.c_[[], []])

        if xs_away[i]:
            away_sc.set_offsets(np.c_[xs_away[i], ys_away[i]])
        else:
            away_sc.set_offsets(np.c_[[], []])

        bx, by = ball_xs[i], ball_ys[i]
        if not (np.isnan(bx) or np.isnan(by)):
            ball_sc.set_offsets(np.c_[[bx], [by]])
        else:
            ball_sc.set_offsets(np.c_[[], []])

        # moving pass line between current passer/receiver positions
        pxy = get_xy(row, passer_id)
        rxy = get_xy(row, receiver_id)
        if not (pd.isna(pxy[0]) or pd.isna(pxy[1]) or pd.isna(rxy[0]) or pd.isna(rxy[1])):
            pass_line.set_data([float(pxy[0]), float(rxy[0])],
                               [float(pxy[1]), float(rxy[1])])
        else:
            pass_line.set_data([], [])

        # texts
        title.set_text(f"Match {match_id} | Period {period_id} | Frame {int(frames_vals[i])}")
        xp = xpass_per_frame[i]
        xpass_text.set_text("xPass: —" if np.isnan(xp) else f"xPass: {xp*100:.1f}%")

        return home_sc, away_sc, ball_sc, pass_line, title, xpass_text, actual_text

    # 0.25x speed → 4× slower: interval base 40ms @25fps → 160ms
    base_interval = 1000 / 25.0
    interval_ms = int(round(base_interval / max(speed_factor, 1e-6)))  # divide by 0.25 -> 160
    ani = FuncAnimation(fig, update, frames=len(win), init_func=init, blit=False, interval=interval_ms, repeat=False)

    # Save (match playback speed in file too)
    target_fps = max(1, int(round(25 * speed_factor)))  # ~6 fps at 0.25x
    if save_path.endswith(".gif"):
        from matplotlib.animation import PillowWriter
        ani.save(save_path, writer=PillowWriter(fps=target_fps))
    else:
        from matplotlib.animation import FFMpegWriter
        ani.save(save_path, writer=FFMpegWriter(fps=target_fps))
    plt.close(fig)
    print(f"Saved animation to {save_path}")


# ---------------- Example usage ----------------
# ---------------- Example usage ----------------
if __name__ == "__main__":
    # 1) Load events & choose a pass
    df_events = pd.read_csv(
        "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_1/Sample_Game_1_RawEventsData.csv"
    ).rename(columns={
        "Type": "event_type", "Subtype": "subtype", "Team": "team_id",
        "Period": "period_id", "Start Frame": "start_frame", "End Frame": "end_frame",
        "From": "player_id", "To": "receiver_player_id"
    })
    df_events["event_type"] = df_events["event_type"].astype(str).str.lower()
    df_passes = df_events[df_events["event_type"] == "pass"].dropna(
        subset=["start_frame", "player_id", "receiver_player_id", "period_id", "end_frame"]
    )

    example = df_passes.iloc[20]
    match_id = 1
    period_id = int(example["period_id"])
    center_frame = int(example["start_frame"])

    # 2) Map event IDs -> tracking IDs (your mapping block you already added)
    tracking = metrica.load_open_data(match_id=match_id, coordinates="metrica")
    home, away = tracking.metadata.teams[0], tracking.metadata.teams[1]
    def get_jersey(p):
        for attr in ("jersey_number", "jersey_no", "number"):
            if hasattr(p, attr):
                v = getattr(p, attr)
                if v is not None and not pd.isna(v):
                    try: return int(v)
                    except Exception: pass
        return None
    def build_player_map(team):
        m = {}
        for p in team.players:
            pid = p.player_id
            m[pid.lower()] = pid
            name = str(getattr(p, "name", "")).strip().lower()
            if name: m[name] = pid
            num = get_jersey(p)
            if num is not None:
                s = str(num)
                m[s] = pid; m[f"player {s}"] = pid; m[f"player{s}"] = pid
        return m
    home_map = build_player_map(home)
    away_map = build_player_map(away)
    team_map = {
        "home": home.team_id, "away": away.team_id,
        str(getattr(home, "name", "home")).strip().lower(): home.team_id,
        str(getattr(away, "name", "away")).strip().lower(): away.team_id,
        "team a": home.team_id, "team b": away.team_id,
    }
    import re
    def candidate_keys(val):
        if pd.isna(val): return []
        s = str(val).strip().lower()
        keys = [s]
        m = re.search(r"\d+", s)
        if m:
            n = m.group(0)
            keys += [n, f"player {n}", f"player{n}"]
        return keys
    def resolve_pid(value, team_raw):
        keys = candidate_keys(value)
        tid = team_map.get(str(team_raw).strip().lower(), None)
        order = [home_map, away_map] if tid == home.team_id else [away_map, home_map]
        for k in keys:
            for mp in order:
                if k in mp: return mp[k]
        return None

    event_team_raw = example["team_id"]
    passer_id = resolve_pid(example["player_id"], event_team_raw)
    receiver_id = resolve_pid(example["receiver_player_id"], event_team_raw)
    if not passer_id or not receiver_id:
        raise RuntimeError("Could not map event player ids to tracking player_ids.")

    # sanity check positions at center frame
    df_tr = tracking.to_df().sort_values(["period_id", "frame_id"])
    center_row = df_tr[(df_tr["period_id"] == period_id) & (df_tr["frame_id"] == center_frame)].iloc[0]
    cpx, cpy = get_xy(center_row, passer_id)
    crx, cry = get_xy(center_row, receiver_id)
    assert not (pd.isna(cpx) or pd.isna(cpy) or pd.isna(crx) or pd.isna(cry)), "Passer/receiver coords are NaN"

    # 3) Actual outcome from events
    actual_success = infer_pass_success_from_events(df_events, example, frame_win=FRAME_WIN)

    # 4) Build model + load weights
    model = build_pass_model(n_players=22, d_player=9, d_global=16)
    model.load_weights("xpass_best_weights.weights.h5")

    # 5) Animate (0.25× speed)
    animate_pass_sequence(
        match_id=match_id,
        period_id=period_id,
        center_frame=center_frame,
        passer_id=passer_id,
        receiver_id=receiver_id,
        model=model,
        save_path="pass_anim.gif",     # use .mp4 if you have ffmpeg
        frames_before=50,
        frames_after=50,
        dynamic=True,
        speed_factor=0.25,
        actual_success=actual_success
    )
