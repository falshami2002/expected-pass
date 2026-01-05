import numpy as np
from typing import Dict, Any
from get_features import (
    nearest_def_to_point,
    defenders_in_radius_point,
    min_lane_block,
    cone_pressure_in_pass_dir,
)


def _as_row(players_xy: np.ndarray, players_speed_dir: np.ndarray | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for i in range(22):
        row[f"p{i}_x"] = float(players_xy[i, 0])
        row[f"p{i}_y"] = float(players_xy[i, 1])
    if players_speed_dir is not None:
        for i in range(22):
            row[f"p{i}_s"] = float(players_speed_dir[i, 0])
            row[f"p{i}_d"] = float(players_speed_dir[i, 1])
    return row


def build_tensors_from_placements(
    players: list[dict],
    ball: dict,
    passer_index: int,
    receiver_index: int,
):
    players_xy = np.array([[p["x"], p["y"]] for p in players], dtype=np.float32)

    bx = float(ball["x"])
    by = float(ball["y"])

    px, py = float(players_xy[passer_index, 0]), float(players_xy[passer_index, 1])
    rx, ry = float(players_xy[receiver_index, 0]), float(players_xy[receiver_index, 1])

    row = _as_row(players_xy)
    row["ball_x"] = bx
    row["ball_y"] = by

    passer_team = players[passer_index]["team"]
    is_attack = np.array([1.0 if p["team"] == passer_team else 0.0 for p in players], dtype=np.float32)
    opp_ids = [f"p{i}" for i in range(22) if players[i]["team"] != passer_team]

    d_player = 9
    feats = np.zeros((22, d_player), dtype=np.float32)
    mask = np.ones((22,), dtype=np.float32)

    for i in range(22):
        x, y = float(players_xy[i, 0]), float(players_xy[i, 1])
        feats[i, 0] = x
        feats[i, 1] = y
        feats[i, 2] = x - px
        feats[i, 3] = y - py
        feats[i, 4] = x - bx
        feats[i, 5] = y - by
        feats[i, 6] = float(is_attack[i])
        feats[i, 7] = 1.0 if i == passer_index else 0.0
        feats[i, 8] = 1.0 if i == receiver_index else 0.0

    pass_distance = float(np.hypot(rx - px, ry - py))
    pass_angle = float(np.arctan2(ry - py, rx - px))
    pass_angle_sin = float(np.sin(pass_angle))
    pass_angle_cos = float(np.cos(pass_angle))

    passer_pid = f"p{passer_index}"
    receiver_pid = f"p{receiver_index}"

    rec_near_def = nearest_def_to_point((rx, ry), opp_ids, row)
    rec_def_5 = defenders_in_radius_point((rx, ry), opp_ids, row, r=0.05)
    lane_min = min_lane_block((px, py), (rx, ry), opp_ids, row)
    passer_near_def = nearest_def_to_point((px, py), opp_ids, row)
    passer_def_r4 = defenders_in_radius_point((px, py), opp_ids, row, r=0.04)
    cone_cnt, cone_min_lat = cone_pressure_in_pass_dir(
        (px, py), (rx, ry), opp_ids, row, max_dist=0.20, half_angle_deg=30
    )

    g = np.array(
        [
            px, py, rx, ry, bx, by,
            pass_distance, pass_angle_sin, pass_angle_cos,
            float(rec_near_def) if np.isfinite(rec_near_def) else 0.0,
            float(rec_def_5) if np.isfinite(rec_def_5) else 0.0,
            float(lane_min) if np.isfinite(lane_min) else 0.0,
            float(passer_near_def) if np.isfinite(passer_near_def) else 0.0,
            float(passer_def_r4) if np.isfinite(passer_def_r4) else 0.0,
            float(cone_cnt) if np.isfinite(cone_cnt) else 0.0,
            float(cone_min_lat) if np.isfinite(cone_min_lat) else 0.0,
        ],
        dtype=np.float32,
    )

    return feats[None, ...], mask[None, ...], g[None, ...]


def predict_xpass(model, players, ball, passer_index: int, receiver_index: int) -> float:
    P, M, G = build_tensors_from_placements(players, ball, passer_index, receiver_index)
    pred = model.predict({"players": P, "players_mask": M, "global_feats": G}, verbose=0)
    return float(pred.ravel()[0])
