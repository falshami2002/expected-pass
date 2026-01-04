import re
import numpy as np
import pandas as pd
from kloppy import metrica
from get_features import *  

FPS = 25
FRAME_WIN = 150  
SEC_WIN = FRAME_WIN / FPS 

# ---------- Utils ----------
def norm_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return (
            df[col].astype("string").fillna("")
              .str.strip().str.lower()
              .str.replace(r"[\s\-]+", "_", regex=True)
        )
    return pd.Series("", index=df.index, dtype="string")

# ---- ID reconciliation for Games 1–2 (CSV -> tracking ids) ----
def reconcile_ids_games12(df_events: pd.DataFrame, tracking) -> pd.DataFrame:
    df = df_events.copy()
    home, away = tracking.metadata.teams[0], tracking.metadata.teams[1]

    team_map = {
        "home": home.team_id,
        "away": away.team_id,
        str(getattr(home, "name", "home")).lower(): home.team_id,
        str(getattr(away, "name", "away")).lower(): away.team_id,
        "team a": home.team_id,
        "team b": away.team_id,
    }
    df["team_id"] = df["team_id"].astype(str).str.strip().str.lower().map(team_map).fillna(df["team_id"])

    def get_jersey(p):
        for attr in ("jersey_number", "jersey_no", "number"):
            if hasattr(p, attr):
                v = getattr(p, attr)
                if v is not None and not pd.isna(v):
                    try:
                        return int(v)
                    except Exception:
                        pass
        return None

    def build_player_map(team):
        m = {}
        for p in team.players:
            pid = p.player_id
            m[pid.lower()] = pid
            pname = str(getattr(p, "name", "")).strip().lower()
            if pname:
                m[pname] = pid
            num = get_jersey(p)
            if num is not None:
                s = str(num)
                m[s] = pid
                m[f"player {s}"] = pid
                m[f"player{s}"] = pid
        return m

    home_map = build_player_map(home)
    away_map = build_player_map(away)

    def candidate_keys(val):
        if pd.isna(val):
            return []
        s = str(val).strip().lower()
        keys = [s]
        m = re.search(r"\d+", s)
        if m:
            n = m.group(0)
            keys += [n, f"player {n}", f"player{n}"]
        return keys

    def resolve_pid(row, col):
        keys = candidate_keys(row.get(col))
        if not keys:
            return np.nan
        t = row.get("team_id")
        order = [home_map, away_map] if t == home.team_id else [away_map, home_map] if t == away.team_id else [home_map, away_map]
        for key in keys:
            for mp in order:
                if key in mp:
                    return mp[key]
        return np.nan

    df["player_id"] = df.apply(lambda r: resolve_pid(r, "player_id"), axis=1)
    df["receiver_player_id"] = df.apply(lambda r: resolve_pid(r, "receiver_player_id"), axis=1)
    return df

# ---- Load events for any of the 3 sample games; compute labels ----
def load_events_any(match_id: int) -> pd.DataFrame:
    if match_id in (1, 2):
        url = f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{match_id}/Sample_Game_{match_id}_RawEventsData.csv"
        df = pd.read_csv(url)

        rename_map = {
            "Type": "event_type",
            "Subtype": "subtype",
            "Team": "team_id",
            "Period": "period_id",
            "Start Frame": "start_frame",
            "Start Time [s]": "timestamp_s",
            "End Frame": "end_frame",
            "End Time [s]": "end_timestamp_s",
            "From": "player_id",
            "To": "receiver_player_id",
            "Start X": "coordinates_x",
            "Start Y": "coordinates_y",
            "End X": "end_coordinates_x",
            "End Y": "end_coordinates_y",
        }
        df = df.rename(columns=rename_map)

        df["event_type_norm"] = norm_col(df, "event_type")
        df["subtype_norm"]    = norm_col(df, "subtype")

        df["timestamp"] = pd.to_timedelta(pd.to_numeric(df.get("timestamp_s"), errors="coerce"), unit="s")
        df["end_timestamp"] = pd.to_timedelta(pd.to_numeric(df.get("end_timestamp_s"), errors="coerce"), unit="s")
        df["start_frame"] = pd.to_numeric(df.get("start_frame"), errors="coerce")
        df["end_frame"] = pd.to_numeric(df.get("end_frame"), errors="coerce")

        df = df.sort_values(["period_id", "timestamp"]).reset_index(drop=True)
        grp = df.groupby("period_id")
        df["next_x"] = grp["coordinates_x"].shift(-1)
        df["next_y"] = grp["coordinates_y"].shift(-1)
        df["next_event_type_norm"] = grp["event_type_norm"].shift(-1)
        df["next_subtype_norm"] = grp["subtype_norm"].shift(-1)
        df["next_start_frame"] = pd.to_numeric(grp["start_frame"].shift(-1), errors="coerce")

        # Label success using frame window
        df["success"] = True
        ONBALL = {"pass", "shot", "dribble", "carry", "recovery"}

        for pid, g in df.groupby("period_id", sort=False):
            sf   = pd.to_numeric(g["start_frame"], errors="coerce").to_numpy()
            ef   = pd.to_numeric(g["end_frame"],   errors="coerce").to_numpy()
            et   = g["event_type_norm"].to_numpy()
            st   = g["subtype_norm"].to_numpy()
            team = g["team_id"].to_numpy()
            idx  = g.index.to_numpy()

            pass_mask = (et == "pass") & ~np.isnan(ef)
            pass_idx  = np.where(pass_mask)[0]
            if pass_idx.size == 0:
                continue

            is_fail_evt = (et == "ball_out") | ((et == "ball_lost") & np.isin(st, ["interception", "head_interception"]))
            is_onball   = np.isin(et, list(ONBALL))

            for i in pass_idx:
                endf = ef[i]
                in_win = (sf > endf) & (sf <= endf + FRAME_WIN)

                if np.any(is_fail_evt & in_win):
                    df.loc[idx[i], "success"] = False
                    continue

                cand = np.where(in_win & is_onball)[0]
                if cand.size:
                    j = cand[0]
                    df.loc[idx[i], "success"] = bool(team[j] == team[i])

        return df

    elif match_id == 3:
        ev = metrica.load_event(
            event_data="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_events.json",
            meta_data="https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml",
            coordinates="metrica",
        )
        df = ev.to_df().copy()

        # Normalize text columns for consistency
        df["event_type_norm"] = norm_col(df, "event_type")
        df["subtype_norm"]    = norm_col(df, "subtype")

        # success is already provided (True/False), so keep it
        assert "success" in df.columns, "Expected 'success' in Game 3 data"

        # Optional: compute next_x/next_y for convenience
        df = df.sort_values(["period_id", "timestamp"]).reset_index(drop=True)
        df["next_x"] = df.groupby("period_id")["coordinates_x"].shift(-1)
        df["next_y"] = df.groupby("period_id")["coordinates_y"].shift(-1)

        return df



    else:
        raise ValueError("Only match_id 1, 2, or 3 are supported.")

# ========== Helpers for NN tensors ==========
def get_all_team_players(tracking):
    home, away = tracking.metadata.teams[0], tracking.metadata.teams[1]
    home_ids = [p.player_id for p in home.players]
    away_ids = [p.player_id for p in away.players]
    return home_ids, away_ids, home, away

def build_player_tensor_for_row(row, passer_pid, receiver_pid, passer_team_id, home_ids, away_ids, px, py, bx, by):
    def fix11(ids):
        ids = list(ids)
        if len(ids) >= 11:
            return ids[:11]
        return ids + [f"PAD_{i}" for i in range(11 - len(ids))]

    if passer_pid in home_ids:
        home_fixed = fix11(home_ids); away_fixed = fix11(away_ids)
        team_order = home_fixed + away_fixed
        team_flag_map = {pid: 1 for pid in home_fixed} | {pid: 0 for pid in away_fixed}
    elif passer_pid in away_ids:
        away_fixed = fix11(away_ids); home_fixed = fix11(home_ids)
        team_order = away_fixed + home_fixed
        team_flag_map = {pid: 1 for pid in away_fixed} | {pid: 0 for pid in home_fixed}
    else:
        home_fixed = fix11(home_ids); away_fixed = fix11(away_ids)
        team_order = home_fixed + away_fixed
        team_flag_map = {pid: 1 if pid in home_fixed else 0 for pid in team_order}

    d_player = 9
    P = len(team_order)  # should be 22
    feats = np.zeros((P, d_player), dtype=np.float32)
    mask  = np.zeros((P,), dtype=np.float32)

    bx = float(bx) if bx is not None and not pd.isna(bx) else np.nan
    by = float(by) if by is not None and not pd.isna(by) else np.nan

    for i, pid in enumerate(team_order):
        if isinstance(pid, str) and pid.startswith("PAD_"):
            # keep zeros, mask stays 0
            continue

        x, y = get_xy(row, pid)

        if pd.isna(x) or pd.isna(y):
            continue

        if pd.isna(px) or pd.isna(py):
            dx_p = 0.0; dy_p = 0.0
        else:
            dx_p = float(x) - float(px)
            dy_p = float(y) - float(py)

        if pd.isna(bx) or pd.isna(by):
            dx_b = 0.0; dy_b = 0.0
        else:
            dx_b = float(x) - float(bx)
            dy_b = float(y) - float(by)

        feats[i, 0] = float(x)
        feats[i, 1] = float(y)
        feats[i, 2] = dx_p
        feats[i, 3] = dy_p
        feats[i, 4] = dx_b
        feats[i, 5] = dy_b
        feats[i, 6] = float(team_flag_map.get(pid, 0))
        feats[i, 7] = 1.0 if pid == passer_pid else 0.0
        feats[i, 8] = 1.0 if (pd.notna(receiver_pid) and pid == receiver_pid) else 0.0

        mask[i] = 1.0

    return feats, mask, team_order

def build_global_features_vector(row_dict):
    px = row_dict["px"]; py = row_dict["py"]
    rx = row_dict["rx"]; ry = row_dict["ry"]
    bx = row_dict["bx"]; by = row_dict["by"]
    pass_distance = row_dict["pass_distance"]
    pass_angle = row_dict["pass_angle"]
    pass_angle_sin = np.sin(pass_angle) if not pd.isna(pass_angle) else 0.0
    pass_angle_cos = np.cos(pass_angle) if not pd.isna(pass_angle) else 1.0

    g = np.array([
        px, py, rx, ry, bx, by,
        pass_distance, pass_angle_sin, pass_angle_cos,
        row_dict["receiver_nearest_def"],
        row_dict["defenders_within_5"],
        row_dict["lane_min_def_dist"],
        row_dict["passer_nearest_def"],
        row_dict["passer_defenders_r4"],
        row_dict["lane_cone_def_cnt"],
        row_dict["lane_cone_min_lat"],
    ], dtype=np.float32)
    g[~np.isfinite(g)] = 0.0
    return g

# ---- Build snapshot features for a given match (CSV + NN tensors) ----
def build_features_for_match(match_id: int):
    df_events = load_events_any(match_id)
    tracking = metrica.load_open_data(match_id=match_id, coordinates="metrica")
    df_tracking = tracking.to_df()

    if match_id in (1, 2):
        df_events = reconcile_ids_games12(df_events, tracking)

    pass_mask = df_events["event_type_norm"].eq("pass") if "event_type_norm" in df_events else df_events["event_type"].astype(str).str.lower().eq("pass")
    df_passes = df_events[pass_mask].copy()
    print(f"[match {match_id}] passes in events: {len(df_passes)}")
    if df_passes.empty:
        return pd.DataFrame(), None

    df_tracking = df_tracking.sort_values(["period_id", "timestamp"])
    if {"ball_x","ball_y"}.issubset(df_tracking.columns):
        df_tracking[["ball_x","ball_y"]] = (
            df_tracking.groupby("period_id")[["ball_x","ball_y"]]
                       .transform(lambda g: g.ffill().bfill())
        )

    snaps = []
    if match_id in (1, 2):
        df_passes["start_frame"] = pd.to_numeric(df_passes.get("start_frame"), errors="coerce")
        for pid, gp in df_passes.groupby("period_id", sort=False):
            gt = df_tracking[df_tracking["period_id"] == pid].copy()
            gt["frame_id"] = pd.to_numeric(gt.get("frame_id"), errors="coerce")
            gp = gp.sort_values("start_frame"); gt = gt.sort_values("frame_id")
            m = pd.merge_asof(
                gp, gt,
                by="period_id",
                left_on="start_frame",
                right_on="frame_id",
                direction="nearest",
                tolerance=2
            )
            snaps.append(m)
    else:
        for pid, gp in df_passes.groupby("period_id", sort=False):
            gt = df_tracking[df_tracking["period_id"] == pid]
            m = pd.merge_asof(
                gp, gt,
                by="period_id",
                left_on="timestamp", right_on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(milliseconds=200)
            )
            snaps.append(m)

    df_pass_snaps = pd.concat(snaps, ignore_index=True) if snaps else pd.DataFrame()
    print(f"[match {match_id}] after merge: {len(df_pass_snaps)}")
    if df_pass_snaps.empty:
        return pd.DataFrame(), None

    home_ids, away_ids, home_obj, away_obj = get_all_team_players(tracking)
    team_players = {home_obj.team_id: home_ids, away_obj.team_id: away_ids}

    mapped_from = df_pass_snaps["player_id"].notna().mean()
    mapped_to   = df_pass_snaps["receiver_player_id"].notna().mean()
    print(f"[match {match_id}] mapped passer: {mapped_from:.1%}, receiver: {mapped_to:.1%}")

    d_missing_passer = d_missing_receiver = kept = 0
    rows = []

    players_tensors = []
    masks_tensors   = []
    globals_feats   = []
    labels          = []

    for _, r in df_pass_snaps.iterrows():
        team = r["team_id"]
        passer = r["player_id"]
        receiver = r["receiver_player_id"]

        px, py = get_xy(r, passer)

        rx, ry = (np.nan, np.nan)
        if pd.notna(receiver):
            rx, ry = get_xy(r, receiver)
        if pd.isna(rx) or pd.isna(ry):
            rx, ry = r.get("end_coordinates_x"), r.get("end_coordinates_y")
        if pd.isna(rx) or pd.isna(ry):
            rx, ry = r.get("next_x"), r.get("next_y")

        bx, by = r.get("ball_x"), r.get("ball_y")

        if pd.isna(px) or pd.isna(py):
            d_missing_passer += 1
            continue
        if pd.isna(rx) or pd.isna(ry):
            d_missing_receiver += 1
            continue

        opp_team = away_obj.team_id if team == home_obj.team_id else home_obj.team_id
        opp_ids = team_players.get(opp_team, [])

        rec_near_def = nearest_def_to_point((rx, ry), opp_ids, r)
        rec_def_5    = defenders_in_radius_point((rx, ry), opp_ids, r, r=0.05)
        lane_min     = min_lane_block((px, py), (rx, ry), opp_ids, r)
        passer_near_def = nearest_def_to_point((px, py), opp_ids, r)
        passer_def_r4   = defenders_in_radius_point((px, py), opp_ids, r, r=0.04)
        cone_cnt, cone_min_lat = cone_pressure_in_pass_dir(
            (px, py), (rx, ry), opp_ids, r, max_dist=0.20, half_angle_deg=30
        )

        row_dict = {
            "match_id": match_id,
            "event_id": r.get("event_id"),
            "period_id": r["period_id"],
            "team_id": team,
            "passer_id": passer,
            "receiver_id": receiver,
            "px": float(px), "py": float(py), "rx": float(rx), "ry": float(ry),
            "bx": float(bx) if bx is not None and not pd.isna(bx) else np.nan,
            "by": float(by) if by is not None and not pd.isna(by) else np.nan,
            "pass_distance": float(np.hypot(float(rx) - float(px), float(ry) - float(py))),
            "pass_angle": float(np.arctan2(float(ry) - float(py), float(rx) - float(px))),
            "receiver_nearest_def": float(rec_near_def) if np.isfinite(rec_near_def) else np.nan,
            "defenders_within_5": float(rec_def_5) if np.isfinite(rec_def_5) else np.nan,
            "lane_min_def_dist": float(lane_min) if np.isfinite(lane_min) else np.nan,
            "passer_nearest_def": float(passer_near_def) if np.isfinite(passer_near_def) else np.nan,
            "passer_defenders_r4": float(passer_def_r4) if np.isfinite(passer_def_r4) else np.nan,
            "lane_cone_def_cnt": float(cone_cnt) if np.isfinite(cone_cnt) else np.nan,
            "lane_cone_min_lat": float(cone_min_lat) if np.isfinite(cone_min_lat) else np.nan,
            "success": r.get("success"),
        }
        rows.append(row_dict)
        kept += 1

        feats_22xD, mask_22, _order = build_player_tensor_for_row(
            r, passer, receiver, team, home_ids, away_ids, px, py, bx, by
        )
        players_tensors.append(feats_22xD)
        masks_tensors.append(mask_22)
        globals_feats.append(build_global_features_vector(row_dict))

        lab = row_dict.get("success")
        if pd.isna(lab):
            # skip unlabeled samples (shouldn't happen now that we label game 3)
            labels.append(np.nan)
        else:
            labels.append(float(lab))

    print(f"[match {match_id}] kept={kept} | dropped: passer={d_missing_passer}, receiver={d_missing_receiver}")

    df_out = pd.DataFrame(rows)

    # Filter out any rows where label is NaN (defensive)
    valid_mask = ~pd.isna(df_out["success"])
    df_out = df_out[valid_mask].reset_index(drop=True)

    if len(df_out) == 0:
        return pd.DataFrame(), None

    # Apply same valid filter to tensors
    valid_idx = np.where(valid_mask.values)[0]
    players_arr = np.stack(players_tensors, axis=0)[valid_idx]
    masks_arr   = np.stack(masks_tensors,   axis=0)[valid_idx]
    globals_arr = np.stack(globals_feats,   axis=0)[valid_idx]
    y_arr       = np.array(labels, dtype=np.float32)[valid_idx]

    nn_payload = {
        "players": players_arr,   # (K, 22, 9)
        "mask":    masks_arr,     # (K, 22)
        "globals": globals_arr,   # (K, 16)
        "y":       y_arr,         # (K,)
    }
    return df_out, nn_payload

# ========== Run for 1,2,3 and save both CSV + NN bundle ==========
if __name__ == "__main__":
    dfs = []
    nn_blobs = []
    for mid in (1, 2, 3):
        print(f"Processing match {mid}...")
        df_m, blob = build_features_for_match(mid)
        if df_m is not None and not df_m.empty:
            print(f"  -> rows: {len(df_m)}, success rate: {df_m['success'].mean():.3f}")
            dfs.append(df_m)
        else:
            print("  -> no rows produced (check ID mapping / merge stage)")
        if blob is not None:
            nn_blobs.append(blob)

    Xy = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    print("Final label counts:", Xy["success"].value_counts(dropna=False) if not Xy.empty else "Empty")
    Xy.to_csv("xpass_training_table_games1_3.csv", index=False)

    if nn_blobs:
        players_np = np.concatenate([b["players"] for b in nn_blobs], axis=0)   # (N, 22, 9)
        mask_np    = np.concatenate([b["mask"]    for b in nn_blobs], axis=0)   # (N, 22)
        globals_np = np.concatenate([b["globals"] for b in nn_blobs], axis=0)   # (N, 16)
        y_np       = np.concatenate([b["y"]       for b in nn_blobs], axis=0)   # (N,)

        meta = {
            "d_player": players_np.shape[-1],
            "n_players": players_np.shape[1],
            "d_global": globals_np.shape[-1],
            "feature_names_player": [
                "x","y","dx_to_passer","dy_to_passer","dx_to_ball","dy_to_ball",
                "team_is_passer_team","is_passer","is_receiver"
            ],
            "feature_names_global": [
                "px","py","rx","ry","bx","by",
                "pass_distance","pass_angle_sin","pass_angle_cos",
                "receiver_nearest_def","defenders_within_5","lane_min_def_dist",
                "passer_nearest_def","passer_defenders_r4","lane_cone_def_cnt","lane_cone_min_lat"
            ],
        }

        np.savez_compressed(
            "xpass_nn_inputs.npz",
            players=players_np,
            mask=mask_np,
            globals=globals_np,
            y=y_np,
            meta=np.array([meta], dtype=object),
        )
        print("Saved NN tensors to xpass_nn_inputs.npz")
        print("Shapes:", players_np.shape, mask_np.shape, globals_np.shape, y_np.shape)
    else:
        print("No NN tensors produced.")
