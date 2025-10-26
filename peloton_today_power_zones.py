#!/usr/bin/env python3
"""
Peloton "Rides for a Date" Graphs

Features:
- --date YYYY-MM-DD  (defaults to "today" in your timezone)
- --two-graphs       (split: power+zones figure AND HR/Cad/Res figure)
- Inline HTML gallery (images embedded; click image to open full-size in a new tab)
- Auto-help (prints usage & timezone help if you run with no args)
- --list-timezones   (lists all IANA timezones on this system and exits)
- --cleanup          (clears the output directory before writing new images/HTML)
- Zone bands always through Z7; normalized y-axis cap so zones look comparable
"""

import argparse
import datetime as dt
import json
import os
import sys
import shutil
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

import requests
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from dateutil import tz

# Python 3.9+ zoneinfo for listing timezones
try:
    from zoneinfo import available_timezones
except Exception:
    available_timezones = None

PELOTON_BASE = "https://api.onepeloton.com"
DEBUG_API = False

# ---------- CLI Helpers ----------
COMMON_TZS = [
    "UTC",
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "America/Phoenix", "America/Anchorage", "Pacific/Honolulu",
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Madrid",
    "Europe/Amsterdam", "Europe/Rome", "Europe/Zurich",
    "Asia/Tokyo", "Asia/Seoul", "Asia/Shanghai", "Asia/Singapore", "Asia/Kolkata",
    "Australia/Sydney", "Australia/Melbourne",
]

def print_extra_help_and_exit(parser: argparse.ArgumentParser, code: int = 0):
    print(parser.format_help())
    print("\nAcceptable timezone examples (IANA):")
    print("  " + ", ".join(COMMON_TZS))
    if available_timezones:
        print("\nRun with --list-timezones to print the full list detected on this system.")
    else:
        print("\nFull timezone listing not available (Python < 3.9).")
    sys.exit(code)

# ---------- Debug JSON helpers ----------
def _truncate_for_debug(obj, max_list=200, max_str=500, _depth=0):
    if _depth > 6:
        return "..."
    if isinstance(obj, dict):
        return {k: _truncate_for_debug(v, max_list, max_str, _depth+1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_list:
            return [_truncate_for_debug(v, max_list, max_str, _depth+1) for v in obj[:max_list]] + [f"... (truncated {len(obj) - max_list} items)"]
        return [_truncate_for_debug(v, max_list, max_str, _depth+1) for v in obj]
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + f"... (truncated {len(obj) - max_str} chars)"
    return obj

def _debug_print(title: str, data):
    if not DEBUG_API:
        return
    try:
        print(f"\n=== Peloton API: {title} ===")
        print(json.dumps(_truncate_for_debug(data), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[DEBUG] Failed to print JSON for {title}: {e}")

# ---------- Data structures ----------
@dataclass
class WorkoutSeries:
    workout_id: str
    title: str
    start_time: int
    duration: int
    metric_slug: str
    seconds: List[float]
    values: List[Optional[float]]

# ---------- Utilities ----------
def _parse_summary_value(summaries: List[dict], slug: str) -> Optional[float]:
    for s in summaries or []:
        if s.get("slug") == slug and isinstance(s.get("value"), (int, float)):
            return float(s["value"])
    return None

def _avg_metric_from_pg(pg: dict, slug: str) -> Optional[float]:
    for m in pg.get("metrics", []):
        if m.get("slug") == slug and isinstance(m.get("average_value"), (int, float)):
            return float(m["average_value"])
    for m in pg.get("metrics", []):
        if m.get("slug") == slug and isinstance(m.get("values"), list) and m.get("values"):
            vals = [v for v in m["values"] if isinstance(v, (int, float))]
            if vals: return sum(vals) / len(vals)
    return None

def _get_metric_series(pg: dict, slug: str) -> Optional[List[Optional[float]]]:
    for m in pg.get("metrics", []):
        if m.get("slug") == slug and isinstance(m.get("values"), list):
            return m.get("values")
    return None

def _infer_every_n_from_seconds(seconds: List[float]) -> int:
    if not seconds or len(seconds) < 3:
        return 5
    diffs = [seconds[i] - seconds[i-1] for i in range(1, len(seconds)) if seconds[i] > seconds[i-1]]
    if not diffs:
        return 5
    try:
        step = statistics.median(diffs)
    except statistics.StatisticsError:
        step = diffs[0]
    return max(1, int(round(step)))

def _compute_normalized_power(values: List[Optional[float]], window_secs: int, every_n: int) -> Optional[float]:
    seq = [v for v in values if isinstance(v, (int, float))]
    if not seq or every_n <= 0:
        return None
    k = max(1, int(round(window_secs / every_n)))
    if len(seq) < k:
        return None
    q, sm, acc = [], 0.0, []
    for v in seq:
        q.append(v); sm += v
        if len(q) > k:
            sm -= q.pop(0)
        if len(q) == k:
            m = sm / k
            acc.append(m**4)
    if not acc:
        return None
    return (sum(acc) / len(acc)) ** 0.25

def _format_hms(seconds: int) -> str:
    m, s = divmod(max(0, int(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def _parse_segments_from_pg(pg: dict, fallback_duration: int) -> Tuple[int, int, int]:
    warm = cool = 0
    segs = pg.get("segment_list") or pg.get("segments") or []
    if isinstance(segs, list) and segs:
        for seg in segs:
            name = (seg.get("name") or seg.get("icon_name") or "").lower()
            slug = (seg.get("icon_slug") or "").lower()
            length = int(seg.get("length") or 0)
            if "warm" in name or slug == "warmup": warm += length
            elif "cool" in name or slug == "cooldown": cool += length
        total = int(pg.get("duration") or fallback_duration or 0)
        return warm, max(0, total - warm - cool), cool
    total = int(pg.get("duration") or fallback_duration or 0)
    return 0, total, 0

# ---------- API ----------
def login(username: str, password: str):
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15)",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": "https://members.onepeloton.com",
    })
    r = s.post(f"{PELOTON_BASE}/auth/login", json={"username_or_email": username, "password": password})
    try:
        data = r.json()
    except Exception:
        data = {"message": r.text}
    _debug_print("/auth/login", data)
    if r.status_code != 200:
        raise RuntimeError(f"Login failed {r.status_code}: {data}")
    return s, data

def get_user_profile(session: requests.Session) -> dict:
    r = session.get(f"{PELOTON_BASE}/api/me")
    r.raise_for_status()
    me = r.json()
    _debug_print("/api/me", me)
    return me

def _fetch_workouts_pages(session: requests.Session, url: str, base_params: dict, max_pages: int = 10):
    results = []
    for page in range(max_pages):
        params = dict(base_params)
        params["page"] = page
        r = session.get(url, params=params)
        if r.status_code in (400, 401, 403, 404, 429, 500):
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise RuntimeError(f"Workouts fetch failed {r.status_code} at {url} page {page}: {detail}")
        data = r.json()
        _debug_print(f"{url}?page={page}", data)
        items = data.get("data") or data.get("workouts") or data.get("items") or []
        if not items:
            break
        results.extend(items)
        if len(items) < base_params.get("limit", 50):
            break
    return results

def _get_ride_details(session: requests.Session, ride_id: str) -> dict:
    for ep in (f"{PELOTON_BASE}/api/ride/{ride_id}", f"{PELOTON_BASE}/api/ride/{ride_id}/details"):
        try:
            r = session.get(ep)
            if r.status_code == 200:
                data = r.json()
                _debug_print(ep, data)
                return data
        except Exception:
            pass
    return {}

def _start_end_epoch_for_date(date_str: Optional[str], tz_str: str) -> Tuple[int, int]:
    tzinfo = tz.gettz(tz_str) or tz.tzlocal()
    if date_str:
        try:
            y, m, d = map(int, date_str.split("-"))
            day_local = dt.datetime(y, m, d, tzinfo=tzinfo)
        except Exception:
            raise ValueError("Invalid --date format. Use YYYY-MM-DD.")
    else:
        day_local = dt.datetime.now(tzinfo)
    start_local = day_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + dt.timedelta(days=1)
    return int(start_local.timestamp()), int(end_local.timestamp())

def get_cycling_workouts_for_date(session: requests.Session, user_id: str, tz_str: str, date_str: Optional[str]):
    start_epoch, end_epoch = _start_end_epoch_for_date(date_str, tz_str)
    base = f"{PELOTON_BASE}/api/user/{user_id}/workouts"
    items = _fetch_workouts_pages(session, base, {"limit": 50, "joins": "ride,ride.instructor"}, max_pages=4) or \
            _fetch_workouts_pages(session, base, {"limit": 50}, max_pages=4)
    filtered = []
    for w in items:
        created = int(w.get("created_at") or w.get("start_time") or 0)
        discipline = (w.get("fitness_discipline") or w.get("sport_type") or "").lower()
        disp = (w.get("fitness_discipline_display_name") or "").lower()
        if start_epoch <= created < end_epoch and (discipline == "cycling" or "ride" in disp):
            if not w.get("ride") and w.get("ride_id"):
                ride = _get_ride_details(session, w["ride_id"]) or {}
                if ride: w["ride"] = ride
            filtered.append(w)
    if DEBUG_API:
        print("\n=== Cycling workouts (date-filtered) ===")
        print(json.dumps(_truncate_for_debug(filtered), indent=2, ensure_ascii=False))
    return filtered

def get_performance_graph(session: requests.Session, workout_id: str) -> dict:
    url = f"{PELOTON_BASE}/api/workout/{workout_id}/performance_graph"
    r = session.get(url, params={"every_n": 5})
    r.raise_for_status()
    data = r.json()
    _debug_print(url, data)
    return data

def detect_ftp_from_pg_meta(pg: dict) -> Optional[float]:
    for k in ["ftp","user_ftp","userFtp","UserFtp","functional_threshold_power"]:
        if k in pg and isinstance(k, str) and isinstance(pg[k], (int, float)):
            return float(pg[k])
    for m in pg.get("metrics", []):
        for k in ["ftp","user_ftp","userFtp","UserFtp","functional_threshold_power"]:
            if k in m and isinstance(m[k], (int, float)):
                return float(m[k])
    return None

def detect_ftp_from_profile(profile: dict) -> Optional[float]:
    for k in ["ftp","user_ftp","default_ftp","functional_threshold_power","functionalThresholdPower","cycling_ftp","cyclingFtp"]:
        v = profile.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    for parent in ["settings","preferences","workout_preferences","cycling"]:
        d = profile.get(parent)
        if isinstance(d, dict):
            for k in ["ftp","user_ftp","default_ftp","functional_threshold_power","cycling_ftp"]:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
    return None

# ---------- Zones & Plotting ----------
def compute_zone_bands(ftp: float) -> List[Tuple[str, float, float]]:
    return [
        ("Z1 (Active Recovery)", 0.00*ftp, 0.55*ftp),
        ("Z2 (Endurance)",       0.56*ftp, 0.75*ftp),
        ("Z3 (Tempo)",           0.76*ftp, 0.90*ftp),
        ("Z4 (Lactate Thresh)",  0.91*ftp, 1.05*ftp),
        ("Z5 (VO2max)",          1.06*ftp, 1.20*ftp),
        ("Z6 (Anaerobic)",       1.21*ftp, 1.50*ftp),
        ("Z7 (Neuromuscular)",   1.51*ftp, float('inf')),
    ]

def _normalized_y_limits_for_zones(ftp: float, signal_max: Optional[float]) -> Tuple[float, float]:
    ymax = max((signal_max or 0) * 1.05, 1.6 * ftp)
    ymin = 0
    return ymin, ymax

def _annotate_header_footer(fig, series: WorkoutSeries, ftp: float, ride_meta: dict, stats: dict):
    title = ride_meta.get("title") or series.title
    instructor = ride_meta.get("instructor") or ""
    pr_flag = ride_meta.get("pr_badge")
    aired = ride_meta.get("original_air_date") or ""
    done_date = ride_meta.get("done_date") or ""
    username_disp = ride_meta.get("username") or ""

    plt.figtext(0.01, 0.98, f"{title} — {instructor}" if instructor else f"{title}",
                ha="left", va="top", fontsize=14, fontweight="bold")
    if pr_flag:
        plt.figtext(0.01, 0.915, "PR", ha="left", va="top", fontsize=11, fontweight="bold",
                    bbox=dict(facecolor="#ffe570", edgecolor="#d4b300", boxstyle="round,pad=0.3"))
    plt.figtext(0.99, 0.98, f"Expected TSS: —", ha="right", va="top", fontsize=12)

    plt.figtext(0.01, 0.94, f"Aired: {aired}", ha="left", va="top", fontsize=11)
    plt.figtext(0.50, 0.94, f"Done: {done_date}    User: {username_disp}", ha="center", va="top", fontsize=11)

    row1 = (
        f"Average Power: {stats.get('avg_output_w','—')} W    "
        f"NP: {stats.get('np_w','—')}    "
        f"Intensity Factor: {stats.get('if_ratio','—')}    "
        f"VI: {stats.get('vi','—')}    "
        f"FTP: {int(round(stats.get('ftp', ftp))) if isinstance(stats.get('ftp', ftp),(int,float)) else '—'}    "
        f"TSS: {stats.get('tss','—')}"
    )
    row2 = (
        f"Cadence: {stats.get('avg_cadence_rpm','—')} rpm    "
        f"Resistance: {stats.get('avg_res_pct','—')} %    "
        f"Calories: {stats.get('calories_kcal','—')} kcal    "
        f"Distance: {stats.get('distance_mi','—')} mi    "
        f"Total Output: {stats.get('total_output_kj','—')} kJ"
    )
    plt.figtext(0.99, 0.94, f"Warm Up: {stats.get('warm_hms','—')}   Main Set: {stats.get('main_hms','—')}   Cool Down: {stats.get('cool_hms','—')}",
                ha="right", va="top", fontsize=11)
    plt.figtext(0.5, 0.045, row1, ha="center", va="bottom", fontsize=11)
    plt.figtext(0.5, 0.018, row2, ha="center", va="bottom", fontsize=11)

def plot_single_graph(series: WorkoutSeries, ftp: float, out_dir: str, ride_meta: dict, stats: dict) -> str:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig = plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")

    try:
        signal_max = max(v for v in series.values if v is not None)
    except ValueError:
        signal_max = 0
    ymin, ymax = _normalized_y_limits_for_zones(ftp, signal_max)

    zones = compute_zone_bands(ftp)
    zone_colors = ["#e6f2ff","#99ccff","#a8ffb5","#fff78a","#ffc37b","#ff9bbd","#ff7c7c"]
    for i, (_, lo, hi) in enumerate(zones):
        hi_capped = min(hi if hi != float('inf') else ymax, ymax)
        ax.axhspan(lo, hi_capped, alpha=0.35, color=zone_colors[i])

    xmins = [s/60.0 for s in series.seconds]
    y = [v if v is not None else float('nan') for v in series.values]
    line_power, = ax.plot(xmins, y, linewidth=2.2, color="#5e35b1", label="Output/Power")

    ax2 = ax.twinx()
    ax2.set_ylabel("Cadence (rpm) / HR (bpm) / Resistance (%)")
    lines_other = []
    for key, style in [("_hr_vals", dict(linewidth=1.2, alpha=0.85, label="Heart Rate")),
                       ("_cad_vals", dict(linestyle="--", linewidth=1.1, alpha=0.9, label="Cadence")),
                       ("_res_vals", dict(linestyle=":", linewidth=1.1, alpha=0.9, label="Resistance"))]:
        vals = ride_meta.get(key)
        if vals:
            l, = ax2.plot(xmins, vals[:len(xmins)], **style); lines_other.append(l)

    ax.set_xlim(left=0); ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel("Time (minutes)", fontweight="bold"); ax.set_ylabel("Watts", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    handles = [line_power] + lines_other
    if handles:
        ax.legend(handles, [h.get_label() for h in handles], loc="upper left", fontsize=9, framealpha=0.9)

    _annotate_header_footer(fig, series, ftp, ride_meta, stats)

    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.fromtimestamp(series.start_time).strftime("%Y%m%d_%H%M")
    out_path = os.path.join(out_dir, f"{ts}_{series.workout_id[:8]}_single.png")
    plt.tight_layout(rect=[0, 0.07, 1, 0.90])
    plt.savefig(out_path, dpi=150); plt.close(fig)
    return out_path

def plot_split_graphs(series: WorkoutSeries, ftp: float, out_dir: str, ride_meta: dict, stats: dict) -> Tuple[str, str]:
    # Power + zones
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig1 = plt.figure(figsize=(14, 6))
    ax1 = plt.gca()
    ax1.set_facecolor("#f9f9f9")

    try:
        signal_max = max(v for v in series.values if v is not None)
    except ValueError:
        signal_max = 0
    ymin, ymax = _normalized_y_limits_for_zones(ftp, signal_max)

    zones = compute_zone_bands(ftp)
    zone_colors = ["#e6f2ff","#99ccff","#a8ffb5","#fff78a","#ffc37b","#ff9bbd","#ff7c7c"]
    for i, (_, lo, hi) in enumerate(zones):
        hi_capped = min(hi if hi != float('inf') else ymax, ymax)
        ax1.axhspan(lo, hi_capped, alpha=0.35, color=zone_colors[i])

    xmins = [s/60.0 for s in series.seconds]
    y = [v if v is not None else float('nan') for v in series.values]
    ax1.plot(xmins, y, linewidth=2.2, color="#5e35b1", label="Output/Power")
    ax1.set_xlim(left=0); ax1.set_ylim(bottom=ymin, top=ymax)
    ax1.set_xlabel("Time (minutes)", fontweight="bold"); ax1.set_ylabel("Watts", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)

    _annotate_header_footer(fig1, series, ftp, ride_meta, stats)
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.fromtimestamp(series.start_time).strftime("%Y%m%d_%H%M")
    out_path_1 = os.path.join(out_dir, f"{ts}_{series.workout_id[:8]}_power.png")
    plt.tight_layout(rect=[0, 0.07, 1, 0.90])
    plt.savefig(out_path_1, dpi=150); plt.close(fig1)

    # HR / Cadence / Resistance
    fig2 = plt.figure(figsize=(14, 5.5))
    ax2 = plt.gca()
    ax2.set_facecolor("#f9f9f9")
    lines = []

    for key, style in [("_hr_vals", dict(linewidth=1.4, alpha=0.95, label="Heart Rate")),
                       ("_cad_vals", dict(linestyle="--", linewidth=1.2, alpha=0.95, label="Cadence")),
                       ("_res_vals", dict(linestyle=":", linewidth=1.2, alpha=0.95, label="Resistance"))]:
        vals = ride_meta.get(key)
        if vals:
            l, = ax2.plot(xmins, vals[:len(xmins)], **style)
            lines.append(l)

    ax2.set_xlim(left=0)
    ax2.set_xlabel("Time (minutes)", fontweight="bold")
    ax2.set_ylabel("HR (bpm) / Cadence (rpm) / Resistance (%)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    if lines:
        ax2.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9, framealpha=0.9)

    title = ride_meta.get("title") or series.title
    instructor = ride_meta.get("instructor") or ""
    plt.figtext(0.01, 0.98, f"{title} — {instructor}" if instructor else f"{title}", ha="left", va="top", fontsize=13, fontweight="bold")

    out_path_2 = os.path.join(out_dir, f"{ts}_{series.workout_id[:8]}_overlays.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(out_path_2, dpi=150); plt.close(fig2)

    return out_path_1, out_path_2

def plot_stacked_graphs(series: WorkoutSeries, ftp: float, out_dir: str, ride_meta: dict, stats: dict, args) -> str:
    """
    One image with two vertically-stacked subplots (dominant top):
      Top:    Heart Rate (left Y) vs Output/Power (right Y) with FTP zone bands
      Bottom: Cadence (left Y), optional Resistance (right Y). No power on bottom.
    Supports HR preprocessing: ignore initial minutes, optional lead/lag shift, optional smoothing.
    """
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, (ax_top_left, ax_bot_left) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}  # top subplot dominant
    )
    fig.patch.set_facecolor("white")

    # -------------------------
    # Timebase & series
    # -------------------------
    xsecs = series.seconds[:]  # seconds from workout start
    xmins = [s / 60.0 for s in xsecs]
    power_vals = [v if v is not None else float('nan') for v in series.values]

    # Optional overlays (already prepared upstream)
    raw_hr  = (ride_meta.get("_hr_vals") or [])[:len(xsecs)]
    cad_vals = (ride_meta.get("_cad_vals") or [])[:len(xsecs)]
    res_vals = (ride_meta.get("_res_vals") or [])[:len(xsecs)]

    # Estimate sampling step (sec)
    if len(xsecs) >= 3:
        steps = [max(1, xsecs[i] - xsecs[i-1]) for i in range(1, len(xsecs))]
        step_sec = sorted(steps)[len(steps)//2]  # median robust
    else:
        step_sec = 1

    # -------------------------
    # HR preprocessing (align & clean)
    # -------------------------
    def _ma(vals, win):
        if win <= 1: return vals
        buf = []
        s = 0.0
        from collections import deque
        q = deque()
        for v in vals:
            if v is None: v = float('nan')
            q.append(v)
            s += v if not math.isnan(v) else 0.0
            if len(q) > win:
                left = q.popleft()
                if not math.isnan(left): s -= left
            # compute average over non-NaNs in window
            non_n = sum(0 if math.isnan(u) else 1 for u in q)
            buf.append((s / non_n) if non_n else float('nan'))
        return buf

    hr_vals = raw_hr[:]

    # 1) Ignore first N minutes of HR
    if args.hr_ignore_min and args.hr_ignore_min > 0:
        n_ignore = int(round((args.hr_ignore_min * 60) / step_sec))
        for i in range(min(n_ignore, len(hr_vals))):
            hr_vals[i] = float('nan')

    # 2) Lead/Lag HR to align with power (positive = shift left/earlier)
    if args.hr_lead_sec and args.hr_lead_sec != 0.0:
        n = int(round(args.hr_lead_sec / step_sec))
        if n > 0:  # lead: drop first n, pad tail
            hr_vals = hr_vals[n:] + [float('nan')] * min(n, len(xsecs))
        elif n < 0:  # lag: pad head, drop tail
            n = abs(n)
            hr_vals = [float('nan')] * min(n, len(xsecs)) + hr_vals[:-n]

    # 3) Smooth HR with moving-average over N seconds (optional)
    if args.hr_smooth_sec and args.hr_smooth_sec > 0:
        import math
        win = max(1, int(round(args.hr_smooth_sec / step_sec)))
        hr_vals = _ma(hr_vals, win)

    # -------------------------
    # TOP: HR (left) vs Output/Power (right) + Zones
    # -------------------------
    ax_hr  = ax_top_left
    ax_pow = ax_hr.twinx()
    ax_hr.set_facecolor("#f9f9f9")

    try:
        signal_max = max(v for v in series.values if v is not None)
    except ValueError:
        signal_max = 0
    ymin, ymax = _normalized_y_limits_for_zones(ftp, signal_max)

    zones = compute_zone_bands(ftp)
    zone_colors = ["#e6f2ff","#99ccff","#a8ffb5","#fff78a","#ffc37b","#ff9bbd","#ff7c7c"]
    for i, (_, lo, hi) in enumerate(zones):
        hi_capped = min(hi if hi != float('inf') else ymax, ymax)
        ax_pow.axhspan(lo, hi_capped, alpha=0.35, color=zone_colors[i], zorder=0)

    # HR in RED; Power in default color
    hr_line = None
    if hr_vals:
        hr_line, = ax_hr.plot(xmins, hr_vals,
                              linewidth=1.8, alpha=0.95, label="Heart Rate",
                              color="#d9534f")  # red tone

    pow_line, = ax_pow.plot(xmins, power_vals,
                            linewidth=2.2, alpha=0.95, label="Output/Power")

    ax_hr.set_ylabel("Heart Rate (bpm)", fontweight="bold")
    ax_pow.set_ylabel("Watts", fontweight="bold")
    ax_hr.grid(True, linestyle="--", alpha=0.35)
    ax_pow.set_ylim(bottom=ymin, top=ymax)
    ax_hr.set_xlim(left=0)

    handles_top = [h for h in [hr_line, pow_line] if h is not None]
    if handles_top:
        ax_hr.legend(handles_top, [h.get_label() for h in handles_top],
                     loc="upper left", fontsize=9, framealpha=0.9)

    # -------------------------
    # BOTTOM: Cadence (left), optional Resistance (right). No power here.
    # -------------------------
    ax_cad = ax_bot_left
    ax_cad.set_facecolor("#f9f9f9")

    cad_line = None
    if cad_vals:
        cad_line, = ax_cad.plot(xmins, cad_vals,
                                linestyle="--", linewidth=1.4, alpha=0.95, label="Cadence")

    res_line = None
    if res_vals:
        ax_res = ax_cad.twinx()
        res_line, = ax_res.plot(xmins, res_vals,
                                linestyle=":", linewidth=1.2, alpha=0.9, label="Resistance")
        ax_res.set_ylabel("Resistance (%)", fontweight="bold")
    else:
        ax_res = None

    ax_cad.set_xlabel("Time (minutes)", fontweight="bold")
    ax_cad.set_ylabel("Cadence (rpm)", fontweight="bold")
    ax_cad.grid(True, linestyle="--", alpha=0.35)

    handles_bot = [h for h in [cad_line, res_line] if h is not None]
    if handles_bot:
        ax_cad.legend(handles_bot, [h.get_label() for h in handles_bot],
                      loc="upper left", fontsize=9, framealpha=0.9)

    # -------------------------
    # Global header/footer
    # -------------------------
    _annotate_header_footer(fig, series, ftp, ride_meta, stats)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    ts = dt.datetime.fromtimestamp(series.start_time).strftime("%Y%m%d_%H%M")
    out_path = os.path.join(out_dir, f"{ts}_{series.workout_id[:8]}_stacked.png")
    plt.tight_layout(rect=[0, 0.07, 1, 0.90])  # leave space for header/footer
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path



# ---------- HTML ----------
def write_inline_gallery(outdir: str, image_paths: List[str], page_title: str, username: str, date_label: str):
    os.makedirs(outdir, exist_ok=True)
    html_path = os.path.join(outdir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<!doctype html><html><head><meta charset='utf-8'>\n")
        f.write(f"<title>{page_title}</title>\n")
        f.write("""
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0b1020;margin:0}
header{padding:1rem 1.25rem;color:#fff;display:flex;align-items:center;gap:1rem}
header h1{font-size:1.1rem;margin:0;color:#cbd5ff}
header .meta{color:#cbd5ff;font-size:.95rem;opacity:.85}
.grid{max-width:1200px;margin:1rem auto 3rem;padding:0 1rem;display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:16px}
.card{background:#11162a;border-radius:16px;padding:.75rem;box-shadow:0 10px 30px rgba(0,0,0,.25)}
.card a{display:block;text-decoration:none}
.card img{width:100%;height:auto;border-radius:12px;display:block;cursor:zoom-in}
.empty{color:#cbd5ff;text-align:center;padding:3rem 1rem;opacity:.8}
</style>
</head><body>
""")
        f.write(f"<header><h1>{page_title}</h1><div class='meta'>User: <strong>{username}</strong> · {date_label}</div></header>\n")
        if image_paths:
            f.write("<div class='grid'>\n")
            for img in image_paths:
                base = os.path.basename(img)
                # Clickable: wrap image in a link to the full-size PNG (opens in new tab)
                f.write(f"  <div class='card'><a href='{base}' target='_blank' rel='noopener'><img src='{base}' alt='Ride graph'></a></div>\n")
            f.write("</div>\n")
        else:
            f.write("<div class='empty'>No cycling workouts found for the selected date.</div>\n")
        f.write("</body></html>")
    return html_path

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Generate Peloton graphs for rides on a given date (default: today).",
        add_help=True
    )
    parser.add_argument("--username", default=os.getenv("PELOTON_USERNAME"), help="Peloton username/email (or set PELOTON_USERNAME)")
    parser.add_argument("--password", default=os.getenv("PELOTON_PASSWORD"), help="Peloton password (or set PELOTON_PASSWORD)")
    parser.add_argument("--tz", default=os.getenv("LOCAL_TZ","America/New_York"), help="IANA timezone (e.g., America/New_York)")
    parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD. Defaults to today in the given timezone.")
    parser.add_argument("--outdir", default="peloton_graphs", help="Output directory for images + index.html")
    parser.add_argument("--two-graphs", action="store_true", help="If set, produce two plots per ride (power/zones AND HR/Cad/Res).")
    parser.add_argument("--cleanup", action="store_true", help="If set, clear the output directory before writing new files.")
    parser.add_argument("--debug-api", action="store_true", help="Print truncated JSON of Peloton API responses (filtered set).")
    parser.add_argument("--list-timezones", action="store_true", help="List available IANA timezones detected on this system and exit.")
    parser.add_argument("--stacked", action="store_true",help="One image with two stacked subplots (top: HR vs Output+Zones; bottom: Cadence vs Output).")
    parser.add_argument("--hr-ignore-min", type=float, default=0.0,
        help="Minutes of heart-rate to ignore/mask from the start (e.g., 2.5)")
    parser.add_argument("--hr-lead-sec", type=float, default=0.0,
        help="Shift HR curve left (lead) by N seconds to align with power; negative to lag")
    parser.add_argument("--hr-smooth-sec", type=float, default=0.0,
        help="Apply a simple moving-average smoothing window to HR, in seconds (e.g., 10)")

    # Auto-print help if no arguments at all
    if len(sys.argv) == 1:
        print_extra_help_and_exit(parser, code=0)

    args = parser.parse_args()

    # Handle timezone listing
    if args.list_timezones:
        if available_timezones:
            tzs = sorted(list(available_timezones()))
            for t in tzs:
                print(t)
        else:
            print("Full timezone listing is unavailable on this Python version. Examples:")
            for t in COMMON_TZS:
                print(t)
        return

    global DEBUG_API
    DEBUG_API = bool(args.debug_api)

    if not args.username:
        print("Error: Provide Peloton username via --username or PELOTON_USERNAME.", file=sys.stderr)
        print_extra_help_and_exit(parser, code=2)
    if not args.password:
        try:
            import getpass
            args.password = getpass.getpass("Peloton password: ")
        except Exception:
            print("Error: Provide Peloton password via --password or PELOTON_PASSWORD.", file=sys.stderr)
            sys.exit(2)

    # Cleanup (if requested)
    if args.cleanup and os.path.isdir(args.outdir):
        shutil.rmtree(args.outdir, ignore_errors=True)
    os.makedirs(args.outdir, exist_ok=True)

    # Login & profile
    session, login_data = login(args.username, args.password)
    me = get_user_profile(session)
    api_username = (me.get("username") or me.get("name") or
                    (login_data.get("user") and (login_data["user"].get("username") or login_data["user"].get("email"))) or
                    args.username)
    user_id = (me.get("id") or me.get("user_id") or
               (login_data.get("user") and login_data["user"].get("id")))

    # Get workouts for the date (or today)
    workouts = get_cycling_workouts_for_date(session, user_id, args.tz, args.date)
    if not workouts:
        print("No cycling workouts found for the selected date.")
        write_inline_gallery(args.outdir, [], "Peloton — Rides", api_username,
                             (args.date or dt.datetime.now().strftime("%Y-%m-%d")))
        return

    ftp_from_profile = detect_ftp_from_profile(me)

    produced_images = []

    for w in workouts:
        workout_id = w["id"]
        ride = w.get("ride") or {}
        if not ride and w.get("ride_id"):
            ride = _get_ride_details(session, w["ride_id"]) or {}
        title = (ride.get("title") or w.get("title") or "Peloton Ride")
        instructor = (ride.get("instructor") or {}).get("name") or (ride.get("instructor") or {}).get("display_name") or ""
        start_time = int(w.get("start_time") or w.get("created_at") or 0)

        pg = get_performance_graph(session, workout_id)

        seconds = pg.get("seconds_since_pedaling_start") or pg.get("seconds_since_start") or []
        every_n = _infer_every_n_from_seconds(seconds)

        duration = int(w.get("duration") or 0)
        if duration <= 0: duration = int(pg.get("duration") or 0)
        if duration <= 0 and seconds: duration = int(round(seconds[-1] - seconds[0]))

        metrics = pg.get("metrics", [])
        power_metric = next((m for m in metrics if m.get("slug") == "power"), None)
        output_metric = next((m for m in metrics if m.get("slug") == "output"), None)
        chosen = power_metric or output_metric or next((m for m in metrics if isinstance(m.get("values"), list)), None)
        if not chosen:
            print(f"Skipping {workout_id}: no numeric series found."); continue

        slug = chosen.get("slug", "output")
        values = (chosen.get("values") or [])
        n = min(len(seconds), len(values)); seconds, values = seconds[:n], values[:n]

        hr_vals = _get_metric_series(pg, "heart_rate")
        cad_vals = _get_metric_series(pg, "cadence")
        res_vals = _get_metric_series(pg, "resistance")

        summaries = pg.get("summaries", [])
        total_output_kj = None
        if isinstance(w.get("total_work"), (int, float)):
            total_output_kj = float(w["total_work"]) / 1000.0
        if total_output_kj is None:
            tw = _parse_summary_value(summaries, "total_work")
            if isinstance(tw, (int, float)): total_output_kj = tw / 1000.0

        calories_kcal = _parse_summary_value(summaries, "calories")
        distance_mi = _parse_summary_value(summaries, "distance")

        avg_output_w = _avg_metric_from_pg(pg, "output")
        if avg_output_w is None and output_metric and isinstance(output_metric.get("average_value"), (int, float)):
            avg_output_w = float(output_metric["average_value"])
        if avg_output_w is None:
            avg_output_w = _avg_metric_from_pg(pg, "power")

        avg_cadence_rpm = _avg_metric_from_pg(pg, "cadence")
        avg_res_pct = _avg_metric_from_pg(pg, "resistance")

        ftp = detect_ftp_from_pg_meta(pg) or ftp_from_profile or 250.0
        np_w = _compute_normalized_power(values, window_secs=30, every_n=every_n)
        if_ratio = tss = vi = None
        if isinstance(np_w, (int,float)) and np_w > 0 and ftp > 0 and duration > 0:
            if_ratio = np_w / ftp
            # TSS = ((Duration(s) * NP^2) / (FTP^2 * 3600)) * 100
            tss = ((duration * (np_w ** 2)) / (ftp ** 2 * 3600)) * 100.0
            if isinstance(avg_output_w, (int,float)) and avg_output_w > 0:
                vi = np_w / avg_output_w

        aired_dt = (ride.get("original_air_time") or ride.get("air_time") or ride.get("scheduled_start_time"))
        original_air_date = dt.datetime.fromtimestamp(int(aired_dt)).strftime("%a %d %b %Y") if aired_dt else ""
        done_date = dt.datetime.fromtimestamp(start_time).strftime("%a %d %b %Y @%I:%M%p %Z")
        warm_s, main_s, cool_s = _parse_segments_from_pg(pg, fallback_duration=duration)

        ride_meta = {
            "title": title, "instructor": instructor,
            "original_air_date": original_air_date, "done_date": done_date,
            "username": api_username, "pr_badge": bool(w.get("is_total_work_personal_record")),
            "_hr_vals": hr_vals, "_cad_vals": cad_vals, "_res_vals": res_vals,
        }
        stats = {
            "total_output_kj": f"{int(round(total_output_kj))}" if isinstance(total_output_kj, (int, float)) else "—",
            "avg_output_w": int(round(avg_output_w)) if isinstance(avg_output_w,(int,float)) else "—",
            "np_w": int(round(np_w)) if isinstance(np_w,(int,float)) else "—",
            "if_ratio": f"{if_ratio:.2f}" if isinstance(if_ratio,(int,float)) else "—",
            "vi": f"{vi:.2f}" if isinstance(vi,(int,float)) else "—",
            "ftp": ftp,
            "tss": f"{tss:.1f}" if isinstance(tss,(int,float)) else "—",
            "avg_cadence_rpm": int(round(avg_cadence_rpm)) if isinstance(avg_cadence_rpm,(int,float)) else "—",
            "avg_res_pct": int(round(avg_res_pct)) if isinstance(avg_res_pct,(int,float)) else "—",
            "calories_kcal": int(round(calories_kcal)) if isinstance(calories_kcal,(int,float)) else "—",
            "distance_mi": f"{distance_mi:.2f}" if isinstance(distance_mi,(int,float)) else "—",
            "warm_hms": _format_hms(warm_s), "main_hms": _format_hms(main_s), "cool_hms": _format_hms(cool_s),
        }

        series = WorkoutSeries(
            workout_id=workout_id, title=title, start_time=start_time,
            duration=duration, metric_slug=slug, seconds=seconds, values=values
        )

        if args.stacked:
            p = plot_stacked_graphs(series, ftp, args.outdir, ride_meta, stats, args)
            produced_images.append(p)
            print(f"Saved: {p}")
        elif args.two_graphs:
            p1, p2 = plot_split_graphs(series, ftp, args.outdir, ride_meta, stats)
            produced_images.extend([p1, p2])
            print(f"Saved: {p1}")
            print(f"Saved: {p2}")
        else:
            p = plot_single_graph(series, ftp, args.outdir, ride_meta, stats)
            produced_images.append(p)
            print(f"Saved: {p}")

    date_label = args.date or dt.datetime.now().strftime("%Y-%m-%d")
    html_path = write_inline_gallery(args.outdir, produced_images, "Peloton — Rides", api_username, date_label)
    print(f"Wrote gallery: {html_path}")

if __name__ == "__main__":
    main()
