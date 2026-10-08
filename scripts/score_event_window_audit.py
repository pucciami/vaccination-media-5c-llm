#!/usr/bin/env python3
"""Score the event-window precision audit.

Input: the final adjudicated labels of the event-window audit
(data/event_window_audit/event_window_audit_export.csv). All 167 model-flagged
articles were coded independently by two annotators blind to the model's
prediction; their disagreements were resolved as for the reference standard
(consensus against the codebook, then adjudication by a third specialist), and
the file holds the resulting labels (columns expert_*). The two independent codings, from which the inter-annotator
agreement was computed, are held by the authors and are not deposited.

Computes, for each flag (Calculation-present; Confidence-undermining), the
post-hoc precision against the adjudicated labels with 95% Wilson score CIs,
matching the CI convention used in the manuscript tables. "Unsure" answers are
reported under two conventions (excluded; counted as not-present) so the
precision estimate is bounded rather than convention-dependent.

Usage:
  python scripts/score_event_window_audit.py <audit.csv>
  python scripts/score_event_window_audit.py            # newest event_window_audit_*.csv in data/event_window_audit/
  python scripts/score_event_window_audit.py <audit.csv> --output-csv out/event_window_audit/audit_scores.csv

With --output-csv, every printed proportion (overall and per event week) is also
written as a table (flag, subset, measure, k, n, proportion, ci_low, ci_high),
which reproduce.py checks against the article.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "event_window_audit"   # deposited audit export lives here


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    if n == 0:
        return "n=0"
    return f"{k}/{n} = {p:.3f} (95% CI {lo:.3f}-{hi:.3f})"


def expert_calculation(r: dict) -> str:
    return (r["expert_calculation"] or "").strip()


def expert_confidence_undermining(r: dict) -> str:
    conf = (r["expert_confidence"] or "").strip()
    if conf in ("", "unsure", "no"):
        return conf
    val = (r["expert_confidence_valence"] or "").strip()
    if not val:
        return ""  # incomplete: present but valence not yet given
    return "yes" if val == "undermining" else "no"


def table_rows(rows: list[dict]) -> list[dict]:
    """The printed proportions as rows: precision (unsure excluded) and, within the audit, the share of
    articles not flagged for an indicator that were positive for it and the share of positives flagged;
    for all audited articles and for each event week."""
    flags = [
        ("calculation_present", "model_calculation_present", expert_calculation),
        ("confidence_undermining", "model_confidence_undermining", expert_confidence_undermining),
    ]
    out = []
    subsets = [("all", rows)] + [(w, [r for r in rows if r["week_start"] == w])
                                 for w in sorted({r["week_start"] for r in rows})]
    for flag, flag_col, expert in flags:
        for subset, sub in subsets:
            flagged = [r for r in sub if r[flag_col].strip().lower() == "true"]
            others = [r for r in sub if r[flag_col].strip().lower() != "true"]
            positives = [r for r in sub if expert(r) == "yes"]
            measures = {
                "precision": (sum(expert(r) == "yes" for r in flagged),
                              sum(expert(r) in ("yes", "no") for r in flagged)),
                "positive_among_unflagged": (sum(expert(r) == "yes" for r in others),
                                             sum(expert(r) in ("yes", "no") for r in others)),
                "flagged_among_positives": (sum(r[flag_col].strip().lower() == "true" for r in positives),
                                            len(positives)),
            }
            for measure, (k, n) in measures.items():
                p, lo, hi = wilson(k, n)
                out.append({"flag": flag, "subset": subset, "measure": measure, "k": k, "n": n,
                            "proportion": p, "ci_low": lo, "ci_high": hi})
    return out


def score(rows: list[dict]) -> None:
    flags = [
        ("Calculation-present", "model_calculation_present", expert_calculation),
        ("Confidence-undermining", "model_confidence_undermining", expert_confidence_undermining),
    ]
    for label, flag_col, expert in flags:
        flagged = [r for r in rows if r[flag_col].strip().lower() == "true"]
        answered = [r for r in flagged if expert(r) in ("yes", "no", "unsure")]
        yes = sum(1 for r in answered if expert(r) == "yes")
        unsure = sum(1 for r in answered if expert(r) == "unsure")
        definite = [r for r in answered if expert(r) in ("yes", "no")]
        print(f"--- {label} ---")
        print(f"  flagged: {len(flagged)}, adjudicated: {len(answered)}, unsure: {unsure}")
        print(f"  precision (unsure excluded):       {fmt(yes, len(definite))}")
        print(f"  precision (unsure = not present):  {fmt(yes, len(answered))}")
        # Within-audit comparison (added 2026-10-05): every audited article was coded for both indicators,
        # so those flagged only for the other indicator are unflagged comparison articles for this one.
        others = [r for r in rows if r[flag_col].strip().lower() != "true"]
        o_definite = [r for r in others if expert(r) in ("yes", "no")]
        o_yes = sum(1 for r in o_definite if expert(r) == "yes")
        positives = [r for r in rows if expert(r) == "yes"]
        p_flagged = sum(1 for r in positives if r[flag_col].strip().lower() == "true")
        print(f"  positive among audited articles not flagged for it: {fmt(o_yes, len(o_definite))}")
        print(f"  flagged among positives found in the audit:         {fmt(p_flagged, len(positives))}")
        print()


def main() -> None:
    args = sys.argv[1:]
    output_csv = None
    if "--output-csv" in args:
        i = args.index("--output-csv")
        output_csv = Path(args[i + 1])
        del args[i:i + 2]
    if args:
        path = Path(args[0])
    else:
        candidates = sorted(EXPORTS.glob("event_window_audit_*.csv")) if EXPORTS.exists() else []
        if not candidates:
            raise SystemExit(f"No export CSV given and none found in {EXPORTS}")
        path = candidates[-1]
    print(f"Scoring {path}\n")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    n_done = sum(1 for r in rows
                 if expert_calculation(r) and expert_confidence_undermining(r))
    print(f"Articles in export: {len(rows)}; fully adjudicated: {n_done}\n")
    score(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table = table_rows(rows)
        with output_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"written: {output_csv}")
    print("Note: report these as post-hoc event-window precision estimates; they do")
    print("not replace the locked gold100 metrics.")


if __name__ == "__main__":
    main()
