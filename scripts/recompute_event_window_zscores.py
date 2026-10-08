"""Recompute the EARS-style weekly z-scores and threshold bands from the deposited series.

Reads the long-format weekly indicator series deposited under ``data/event_window/`` and, for
each of the 13 indicators, standardises every week against the seven-week pre-event baseline
(mean and sample SD of the weeks flagged ``is_baseline``), then assigns the one-sided
prediction-limit band. Thresholds are t(6) quantiles inflated by sqrt(1 + 1/7) for a single
new observation against a 7-point baseline.

Usage (from the package root):
    python scripts/recompute_event_window_zscores.py \
        [--input data/event_window/longitudinal_2021_zscores_all_indicators_v3.csv] \
        [--output-dir out/event_window]

The one-sided p-value of a week is the upper tail of t(6) at z / sqrt(1 + 1/7), the same
distribution as the limits; the Bonferroni-adjusted value multiplies it by the 26 screened
indicator-weeks (13 indicators x 2 event weeks), capped at 1.

Outputs: ``event_window_zscores_recomputed.csv`` (all rows, recomputed z, band and one-sided
p-value, plus the deposited ``z_ears`` for comparison), ``event_window_limits.csv`` (the
prediction limits in baseline SD units) and a printed summary of the event weeks, the
out-of-window alarm count, and the Bonferroni-corrected limit for the 26 screened
indicator-weeks. Dependencies: pandas only (the t(6) distribution is computed in closed form).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

N_SCREENED = 26                       # 13 indicators x 2 event weeks
INFL = math.sqrt(1 + 1 / 7)          # new-observation inflation for a 7-week baseline


def t6_sf(t: float) -> float:
    """Upper tail P(T > t) of Student's t with 6 degrees of freedom (closed form for even df)."""
    theta = math.atan(t / math.sqrt(6))
    c2 = math.cos(theta) ** 2
    return 0.5 - 0.5 * math.sin(theta) * (1 + c2 / 2 + 3 * c2 * c2 / 8)


def t6_upper_quantile(p: float) -> float:
    """t such that P(T > t) = 1 - p, by bisection on t6_sf (R: qt(p, 6))."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t6_sf(mid) > 1 - p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# One-sided upper quantiles of t(6); tabulated R values qt(p, 6) as a cross-check.
T6 = {p: t6_upper_quantile(p) for p in (0.95, 0.99, 0.999)}
assert all(abs(T6[p] - v) < 1e-5 for p, v in {0.95: 1.943180, 0.99: 3.142668, 0.999: 5.207626}.items())
LIMIT = {k: v * INFL for k, v in T6.items()}   # 2.077 / 3.360 / 5.567
BONF_26 = t6_upper_quantile(1 - 0.05 / N_SCREENED) * INFL   # 4.876, one-sided, 26 indicator-weeks


def p_one_sided(z: float) -> float:
    """One-sided p-value of a week: upper tail of t(6) at z / sqrt(1 + 1/7)."""
    return t6_sf(z / INFL)


def band(z: float) -> str:
    if z >= LIMIT[0.999]:
        return "strong_signal_above_999"
    if z >= LIMIT[0.99]:
        return "signal_99to999"
    if z >= LIMIT[0.95]:
        return "screening_95to99"
    return "within_reference"


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "t", "yes"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--input", type=Path, default=root / "data" / "event_window" / "longitudinal_2021_zscores_all_indicators_v3.csv")
    ap.add_argument("--output-dir", type=Path, default=root / "out" / "event_window")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df["period"] = pd.to_datetime(df["period"])
    df["_base"] = truthy(df["is_baseline"])
    df["_sig"] = truthy(df["is_signal_window"])
    df["_ctx"] = truthy(df["is_context_week"]) if "is_context_week" in df.columns else False
    df["value"] = df["value"].astype(float)

    out = []
    for ind, g in df.groupby("indicator_id", sort=False):
        base = g.loc[g["_base"], "value"]
        mu, sd = float(base.mean()), float(base.std(ddof=1))
        z = (g["value"] - mu) / sd
        p = z.map(p_one_sided)
        gg = g.assign(baseline_mean=mu, baseline_sd=sd, z_recomputed=z, band_recomputed=z.map(band),
                      p_one_sided=p, p_bonferroni_26=(p * N_SCREENED).clip(upper=1.0))
        out.append(gg)
    res = pd.concat(out).sort_values(["indicator_id", "period"])
    keep = ["period", "indicator_id", "family", "value", "is_baseline", "is_signal_window"] + (["is_context_week"] if "is_context_week" in df.columns else []) + ["baseline_mean", "baseline_sd", "z_recomputed", "band_recomputed", "p_one_sided", "p_bonferroni_26"] + (["z_ears", "ears_band"] if "z_ears" in df.columns else [])
    res[keep].to_csv(args.output_dir / "event_window_zscores_recomputed.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        [{"limit": f"one_sided_{int(k * 1000) / 10:g}", "t6_quantile": T6[k], "sd_units": LIMIT[k]} for k in T6]
        + [{"limit": "bonferroni_26", "t6_quantile": BONF_26 / INFL, "sd_units": BONF_26}]
    ).to_csv(args.output_dir / "event_window_limits.csv", index=False, encoding="utf-8")

    weeks = sorted(df["period"].unique())
    print(f"series: {len(weeks)} weeks, {weeks[0].date()} to {weeks[-1].date()}; {df['indicator_id'].nunique()} indicators")
    print(f"limits (baseline SD units): 95% {LIMIT[0.95]:.3f}, 99% {LIMIT[0.99]:.3f}, 99.9% {LIMIT[0.999]:.3f}; Bonferroni-26 {BONF_26:.3f}")
    if "z_ears" in res.columns:
        dev = (res["z_recomputed"] - res["z_ears"].astype(float)).abs().max()
        same = (res["band_recomputed"] == res["ears_band"]).mean()
        print(f"agreement with deposited z_ears: max |dz| = {dev:.4f}; identical band in {same:.1%} of rows")

    sig = res[res["_sig"]].sort_values(["period", "z_recomputed"], ascending=[True, False])
    for p, g in sig.groupby("period"):
        above = g[g["z_recomputed"] >= LIMIT[0.95]]
        bonf = g[g["z_recomputed"] >= BONF_26]
        print(f"event week {p.date()}: {len(above)} of {len(g)} indicators above the 95% limit; "
              f"above Bonferroni-26: {', '.join(f'{r.indicator_id} (z {r.z_recomputed:.2f}, one-sided p {r.p_one_sided:.4f}, Bonferroni-adjusted {r.p_bonferroni_26:.3f})' for r in bonf.itertuples()) or 'none'}")

    oow = res[~res["_base"] & ~res["_sig"] & ~res["_ctx"]]
    n_alarm = int((oow["z_recomputed"] >= LIMIT[0.95]).sum())
    print(f"out-of-window indicator-weeks: {len(oow)} ({oow['period'].nunique()} weeks x {oow['indicator_id'].nunique()} indicators); "
          f"above the 95% limit: {n_alarm} ({n_alarm / len(oow):.1%})")
    for r in oow[oow["z_recomputed"] >= LIMIT[0.95]].sort_values("z_recomputed", ascending=False).itertuples():
        print(f"   {r.period.date()}  {r.indicator_id}  z = {r.z_recomputed:.2f}")
    print(f"written: {args.output_dir / 'event_window_zscores_recomputed.csv'}")


if __name__ == "__main__":
    main()
