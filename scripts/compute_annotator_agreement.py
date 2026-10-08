#!/usr/bin/env python
"""Inter-annotator agreement before adjudication, with bootstrap confidence intervals.

Computes Supplementary Table S2 from the two annotators' labels for the 400 annotated articles
(data/annotator_agreement/annotator_labels_400.csv, label fields only, no article text):

- relevance, over all 400 articles;
- primary disease and any-5C presence, over the articles both annotators marked relevant;
- presence of each construct, over the articles both marked as having 5C content;
- valence of each construct (three classes), over the articles in which both marked it present.

A missing value is dropped pairwise. Cohen's kappa is computed directly, without scikit-learn, and
reported as undefined when chance agreement is 1 (both annotators used one category throughout).
Intervals are article-level percentile bootstraps (10,000 resamples, seed 20260527); resamples in
which kappa is undefined are skipped and counted in the column defined_resamples.

Usage, from the repository root:
    python scripts/compute_annotator_agreement.py --output-dir out/annotator_agreement
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "annotator_agreement" / "annotator_labels_400.csv"

CONSTRUCTS = ["confidence", "complacency", "constraints", "calculation",
              "collective_responsibility"]
LABEL = {"confidence": "Confidence", "complacency": "Complacency",
         "constraints": "Constraints", "calculation": "Calculation",
         "collective_responsibility": "Collective Responsibility"}


def valence_cols(construct: str) -> tuple[str, str]:
    stem = "collective" if construct == "collective_responsibility" else construct
    return f"ann1_valence_{stem}_d1", f"ann2_valence_{stem}_d1"


def parse_constructs(cell: str | None) -> list[str]:
    """`c5_presence_d1` is a JSON array, a bare token, or empty."""
    if cell is None or cell == "":
        return []
    if cell.startswith("["):
        try:
            return list(json.loads(cell))
        except json.JSONDecodeError:
            return []
    return [cell]


def cohen_kappa(pairs: list[tuple]) -> float | None:
    """Cohen's kappa, or None when it is undefined.

    Kappa is undefined when there are no pairs or when expected (chance) agreement is 1, i.e.
    both annotators used one and the same single category throughout: (po - pe) / (1 - pe) is
    then 0/0. Such a sample carries no information on agreement beyond chance, so it is reported
    as undefined rather than as perfect agreement.
    """
    n = len(pairs)
    if n == 0:
        return None
    labels = sorted({v for p in pairs for v in p}, key=str)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == lab) / n) *
             (sum(1 for _, b in pairs if b == lab) / n) for lab in labels)
    if abs(1.0 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def percent_agreement(pairs: list[tuple]) -> float | None:
    return sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else None


def bootstrap_ci(pairs: list[tuple], reps: int, seed: int, alpha: float = 0.05):
    """Article-level percentile bootstrap. Returns (lo, hi, n_defined).

    Resamples in which kappa is undefined (see ``cohen_kappa``) are skipped and counted; the
    percentiles are taken over the defined resamples only. If fewer than half the resamples are
    defined, no interval is returned.
    """
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(reps):
        k = cohen_kappa([pairs[rng.randrange(n)] for _ in range(n)])
        if k is not None:
            vals.append(k)
    if len(vals) < reps * 0.5:
        return None, None, len(vals)
    vals.sort()
    return (vals[int((alpha / 2) * len(vals))],
            vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))],
            len(vals))


def build(rows: list[dict]) -> list[tuple[str, str, list[tuple]]]:
    """Reproduce each statistic's pair set and denominator, in report order.

    Missing values are dropped pairwise, matching the original ``compute()``.
    """
    def pairs(subset, c1, c2):
        return [(r[c1], r[c2]) for r in subset if r.get(c1) and r.get(c2)]

    out: list[tuple[str, str, list[tuple]]] = []
    out.append(("Relevance", "relevance",
                pairs(rows, "ann1_relevance", "ann2_relevance")))

    rel = [r for r in rows
           if r.get("ann1_relevance") == "relevant" and r.get("ann2_relevance") == "relevant"]
    out.append(("Primary disease", "disease",
                pairs(rel, "ann1_disease_1", "ann2_disease_1")))
    out.append(("Any-5C presence", "has_relevant_5c",
                pairs(rel, "ann1_has_relevant_5c", "ann2_has_relevant_5c")))

    both5c = [r for r in rel
              if r.get("ann1_has_relevant_5c") == "yes" and r.get("ann2_has_relevant_5c") == "yes"]
    for r in both5c:
        r["_a1"] = parse_constructs(r.get("ann1_c5_presence_d1"))
        r["_a2"] = parse_constructs(r.get("ann2_c5_presence_d1"))

    for c in CONSTRUCTS:  # presence is a boolean per article; never missing
        out.append(("Construct presence", LABEL[c],
                    [(c in r["_a1"], c in r["_a2"]) for r in both5c]))

    for c in CONSTRUCTS:  # valence only where both marked the construct present
        c1, c2 = valence_cols(c)
        present = [r for r in both5c if c in r["_a1"] and c in r["_a2"]]
        out.append(("Construct valence", LABEL[c], pairs(present, c1, c2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", "--out", dest="out", type=Path,
                    default=ROOT / "out" / "annotator_agreement")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260527)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input not found: {args.input}\n"
                         "Pass --input with the path to the annotator label file.")
    with args.input.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"articles: {len(rows)}   input: {args.input.name}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"{'level':<20} {'field':<26} {'n':>4} {'% agr':>6} {'kappa':>7}  95% CI")
    print("-" * 82)
    for level, field, pr in build(rows):
        k = cohen_kappa(pr)
        pa = percent_agreement(pr)
        lo, hi, ndef = bootstrap_ci(pr, args.reps, args.seed)
        ci = f"{lo:.3f}\u2013{hi:.3f}" if lo is not None else "uninformative"
        note = ""
        if lo is not None and hi - lo > 0.6:
            note = "  <- interval spans most of the scale"
        if lo is not None and lo == hi:
            note = "  <- degenerate (no disagreement to resample)"
        if ndef < args.reps:
            note += f"  [{args.reps - ndef} of {args.reps} resamples undefined, skipped]"
        kappa_text = f"{k:7.3f}" if k is not None else "  undef"
        print(f"{level:<20} {field:<26} {len(pr):>4} {pa:6.2f} {kappa_text}  {ci}{note}")
        records.append({"level": level, "field": field, "n": len(pr),
                        "percent_agreement": round(pa, 4),
                        "kappa": round(k, 6) if k is not None else "",
                        "ci_low": round(lo, 6) if lo is not None else "",
                        "ci_high": round(hi, 6) if hi is not None else "",
                        "method": "article_level_percentile_bootstrap",
                        "bootstrap_reps": args.reps, "seed": args.seed,
                        "defined_resamples": ndef})

    dest = args.out / "annotator_agreement_with_cis.csv"
    with dest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
