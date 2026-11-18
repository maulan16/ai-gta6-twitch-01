# ai-gta6-twitch-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maulan16/ai-gta6-twitch-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/maulan16/ai-gta6-twitch-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers, hour by hour, as five quantiles (5%, 25%, 50%, 75%, 95%) for hours 2–24 of 19 November 2026 (UTC), from the first hour's print alone.

## Model

GTA VI is console-only with a rolling midnight-local unlock (New Zealand first, then Europe, the UK, and the US), so its Twitch curve should dip after the hour-1 midnight spike, then rise through the UTC day as regions unlock and peak in US evening hours. The program fits that shape at run time from the case's own comparables:

- **Training set:** the case's comparables that are clean console launches (tier "comparable", no PC in platforms, a full 24-hour release day). PC-simultaneous and early-access-contaminated launches are excluded — they front-load hour 1 and stay flat, the wrong shape for a console-only rolling launch.
- **Per-hour OLS regression** on the log scale: `log(v_h) = α_h + β_h · log(v_1)`, fit on the clean console family. The slope falls from ~1.0 in the early hours to ~0.65 late (a big midnight launch is already near the day's ceiling; a small one has headroom to rise).
- **Quantiles:** `q_τ(h) = exp(α_h + β_h · log(p1) + z_τ · sd_h · 1.25)` with normal quantiles `z`, `sd_h` the per-hour residual standard error (n−2 degrees of freedom for the two fitted parameters), and a 1.25 spread widening that fixed under-coverage on validation.
- **Fallbacks:** fewer than 5 clean console comparables → all comparable-tier → all full days → the house baseline shape.

Python 3.13, standard library only, deterministic (two runs on the same case write identical output), no network, no randomness, no clock.

## Validation (practice cases)

Mean pinball loss on the log scale, 4 practice cases (Marvel's Wolverine, Red Dead Redemption 2, God of War Ragnarök, Marvel's Spider-Man 2; true hours 2–24 reconstructed from the kit's documented source, SullyGnome, whose hour-1 prints match the kit exactly):

| model | mean pinball loss |
|---|---|
| this forecaster | **0.192** |
| house baseline | 0.387 |

Leave-one-out over the 11 clean console launches: 0.174 vs baseline 0.359.

## Run it

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/path/to/output python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` with 23 rows (`hour` 2–24, `q` = [q05, q25, q50, q75, q95]).

---

<sub>Entered on [make.fun](https://make.fun/?ref=64vgw5ye&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
