import numpy as np
import pandas as pd

def get_xy(row, pid):
    return row.get(f"{pid}_x"), row.get(f"{pid}_y")

def get_vs(row, pid):
    s = row.get(f"{pid}_s"); d = row.get(f"{pid}_d")
    if pd.isna(s) or pd.isna(d): 
        return None, None
    return float(s*np.cos(d)), float(s*np.sin(d))

def nearest_def_to_point(point_xy, opp_ids, row):
    px, py = point_xy
    d = []
    for pid in opp_ids:
        x, y = get_xy(row, pid)
        if pd.notna(x) and pd.notna(y):
            d.append(np.hypot(px-x, py-y))
    return float(min(d)) if d else np.nan

def defenders_in_radius_point(point_xy, opp_ids, row, r=0.04):
    px, py = point_xy
    c = 0
    for pid in opp_ids:
        x, y = get_xy(row, pid)
        if pd.notna(x) and pd.notna(y) and np.hypot(px-x, py-y) <= r:
            c += 1
    return int(c)

def closing_speed_toward_point(point_xy, opp_ids, row):
    px, py = point_xy
    vals = []
    for pid in opp_ids:
        x, y = get_xy(row, pid)
        vx, vy = get_vs(row, pid)
        if pd.isna(x) or pd.isna(y) or vx is None: 
            continue
        rel = np.array([px-x, py-y])
        dist = np.linalg.norm(rel) + 1e-9
        dir_to_point = rel / dist
        v_close = float(vx*dir_to_point[0] + vy*dir_to_point[1])
        vals.append(v_close)
    return float(max(vals)) if vals else np.nan

def time_to_reach_point(point_xy, opp_ids, row, eps=1e-3):
    px, py = point_xy
    times = []
    for pid in opp_ids:
        x, y = get_xy(row, pid)
        vx, vy = get_vs(row, pid)
        if pd.isna(x) or pd.isna(y) or vx is None:
            continue
        dist = np.hypot(px-x, py-y)
        speed = np.hypot(vx, vy)
        if speed > eps:
            times.append(dist / speed)
    return float(min(times)) if times else np.nan

def cone_pressure_in_pass_dir(passer_xy, receiver_xy, opp_ids, row, max_dist=0.20, half_angle_deg=30):
    ax, ay = passer_xy; bx, by = receiver_xy
    v = np.array([bx-ax, by-ay]); v_norm = np.linalg.norm(v) + 1e-9
    u = v / v_norm
    cos_th = np.cos(np.deg2rad(half_angle_deg))
    cnt = 0
    min_lane_dist = []
    for pid in opp_ids:
        x, y = get_xy(row, pid)
        if pd.isna(x) or pd.isna(y): 
            continue
        w = np.array([x-ax, y-ay])
        proj = float(np.dot(w, u))              
        if 0 <= proj <= max_dist:
            w_norm = np.linalg.norm(w) + 1e-9
            cos_ang = float(np.dot(w, u) / w_norm)  
            if cos_ang >= cos_th:
                cnt += 1
                lat = float(np.linalg.norm(w - proj*u))
                min_lane_dist.append(lat)
    return int(cnt), (min(min_lane_dist) if min_lane_dist else np.nan)

def seg_point_min_dist(ax, ay, bx, by, px, py):
    abx, aby = bx-ax, by-ay
    apx, apy = px-ax, py-ay
    ab2 = abx*abx + aby*aby + 1e-9
    t = max(0.0, min(1.0, (apx*abx + apy*aby) / ab2))
    qx, qy = ax + t*abx, ay + t*aby
    return float(np.hypot(px - qx, py - qy))

def min_lane_block(passer_xy, receiver_xy, opp_ids, row):
    ax, ay = passer_xy; bx, by = receiver_xy
    vals = []
    for pid in opp_ids:
        x, y = get_xy(row, pid)
        if pd.notna(x) and pd.notna(y):
            vals.append(seg_point_min_dist(ax, ay, bx, by, x, y))
    return float(min(vals)) if vals else np.nan
