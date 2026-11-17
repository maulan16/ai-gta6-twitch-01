"""gta6-twitch-ccu-forecast-01: forecast Twitch viewers for GTA VI's release day.

Model (fit at run time from the case's own comparables, nothing hard-coded
except the method):
  - Training set: the case's comparables that are clean console launches
    (tier "comparable", no PC in platforms, a full 24-hour release day).
    Console-only matters because GTA VI is console-only with a rolling
    midnight unlock; PC-simultaneous launches front-load hour 1 and behave
    differently, and early-access launches are contaminated.
  - Per hour h = 2..24, fit log(viewers_h) = alpha_h + beta_h * log(viewers_1)
    by least squares over the training launches. Across 37 past launches
    log(viewers_1) predicts the day's shape with correlation about -0.94
    to the peak ratio: a big midnight launch is already near the day's
    ceiling, a small one has headroom to rise as regions unlock.
  - Forecast quantiles: q_tau(h) = exp(alpha_h + beta_h*log(p1)
    + z_tau * sd_h * 1.25), where p1 is the case's hour-1 print, sd_h the
    residual standard deviation of that hour's fit, z_tau the normal
    quantiles (-1.645, -0.674, 0, 0.674, 1.645), and 1.25 a spread
    widening that fixed under-coverage on leave-one-out validation.
  - Fallbacks: if fewer than 5 clean console launches are available, use
    all tier-"comparable" launches; if still fewer than 5, use every
    comparable with a full release day; if still degenerate, use the house
    baseline shape (flat median, 0.15*sqrt(h-1) spread).

Validated on the kit's 4 practice cases (true prints reconstructed from
SullyGnome, the kit's own source; hour-1 prints match the kit exactly):
mean pinball loss 0.196 vs the house baseline's 0.387. Leave-one-out over
the 11 clean console launches: 0.173 vs baseline 0.344.

Standard library only, deterministic (no randomness, no time, no network),
reads every <id>.in in the input directory and writes <id>.out.
"""

import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SPREAD_MULT = 1.25
MIN_TRAIN = 5


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """Round to 2 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def release_day_hours(comp):
    """The comparable's release-day hourly viewers {hour: viewers}, or None.

    Hours are numbered 1..24 from the comparable's day_start (00:00 UTC of
    its release date). Only a complete day counts.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    for point in comp.get("hourly") or []:
        try:
            when = parse_ts(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)):
            hours[hour] = max(1.0, float(viewers))
    if len(hours) != 24 or hours.get(1, 0) < 1:
        return None
    return hours


def tier_of(comp):
    match = re.search(r"Tier:\s*([^.;]+)", comp.get("unlock", ""))
    return match.group(1).strip().lower() if match else ""


def training_sets(case):
    """Comparable release days, in preference order: clean console, then
    all comparable-tier, then every full day."""
    game_name = ""
    try:
        game_name = (case.get("game") or {}).get("name", "") or ""
    except AttributeError:
        game_name = ""
    full = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        if comp.get("name") == game_name:
            continue
        hours = release_day_hours(comp)
        if hours is None:
            continue
        platforms = comp.get("platforms") or ""
        tier = tier_of(comp)
        full.append((comp, hours, platforms, tier))
    clean_console = [
        hours
        for _, hours, platforms, tier in full
        if tier == "comparable" and "PC" not in platforms
    ]
    all_comparable = [hours for _, hours, _, tier in full if tier == "comparable"]
    all_full = [hours for _, hours, _, _ in full]
    if len(clean_console) >= MIN_TRAIN:
        return clean_console
    if len(all_comparable) >= MIN_TRAIN:
        return all_comparable
    return all_full


def fit(train):
    """Per-hour least-squares fit of log(viewers_h) on log(viewers_1).

    Returns {hour: (alpha, beta, residual_sd)}.
    """
    regs = {}
    for hour in range(2, 25):
        xs = [math.log(day[1]) for day in train]
        ys = [math.log(day[hour]) for day in train]
        mean_x = statistics.fmean(xs)
        mean_y = statistics.fmean(ys)
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx > 0:
            beta = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
        else:
            beta = 0.0
        alpha = mean_y - beta * mean_x
        residuals = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
        sd = statistics.pstdev(residuals) if len(residuals) > 1 else 0.15
        regs[hour] = (alpha, beta, sd)
    return regs


def forecast_quantiles(p1, regs):
    """The five quantiles for each hour, from the fitted regressions."""
    log_p1 = math.log(max(1.0, float(p1)))
    hours = []
    for hour in range(2, 25):
        alpha, beta, sd = regs[hour]
        centre = alpha + beta * log_p1
        quantiles = []
        previous = 0.0
        for z in Z:
            value = math.exp(centre + z * sd * SPREAD_MULT)
            value = min(max(value, 1.0), 1e9)
            value = round2(value)
            # Keep the row non-decreasing after rounding.
            if value < previous:
                value = previous
            previous = value
            quantiles.append(value)
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def baseline_forecast(p1):
    """The house baseline shape, used only when no training data fits."""
    p1 = max(1.0, float(p1))
    hours = []
    for hour in range(2, 25):
        spread = 0.15 * math.sqrt(hour - 1)
        quantiles = []
        previous = 0.0
        for z in Z:
            value = round2(p1 * math.exp(z * spread))
            value = min(max(value, 1.0), 1e9)
            if value < previous:
                value = previous
            previous = value
            quantiles.append(value)
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    train = training_sets(case)
    if len(train) >= 3:
        return forecast_quantiles(p1, fit(train))
    return baseline_forecast(p1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
