import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


CONSTRUCTS = [
    "confidence",
    "complacency",
    "constraints",
    "calculation",
    "collective_responsibility",
]

VALENCE_FIELDS = {
    "confidence": "valence_confidence_d1",
    "complacency": "valence_complacency_d1",
    "constraints": "valence_constraints_d1",
    "calculation": "valence_calculation_d1",
    "collective_responsibility": "valence_collective_d1",
}

# Defaults resolve against the deposited inputs in data/locked_set_comparisons/,
# so the script runs from the package root with no arguments. These are the
# standard-F1 comparison files behind the published Tables 1-2 and abstract.
_DEPOSIT_COMPARISONS = "data/locked_set_comparisons"

def _stable_seed(base_seed: int, label: str) -> int:
    """Deterministic per-unit seed.

    Seeds were previously drawn sequentially from one generator over the set of
    discovered comparison files, so an interval depended on which other systems
    happened to be in the run: the same system gave different bounds in a full
    run and in a subset. Deriving the seed from the base seed and the unit's own
    identifier makes every interval reproducible on its own.
    """
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")

DEFAULT_MATRIX_DIRS = [_DEPOSIT_COMPARISONS]


def _cmp(system_id: str) -> str:
    return f"{system_id}={_DEPOSIT_COMPARISONS}/{system_id}/{system_id}_comparison.csv"


DEFAULT_PAIRED_COMPARISONS = [
    f"{_cmp('G0_E2')}::{_cmp('G0_E0')}",
    f"{_cmp('G0_E2')}::{_cmp('G0_E1')}",
    f"{_cmp('G0_E2')}::{_cmp('A0_F2')}",
    # Adaptation steps within each family (Supplementary Table S1). Each comparison draws its
    # own seed from its name (_stable_seed), so adding comparisons leaves the others unchanged.
    f"{_cmp('G0_E1')}::{_cmp('G0_E0')}",
    f"{_cmp('A0_F1')}::{_cmp('A0_F0')}",
    f"{_cmp('A0_F2')}::{_cmp('A0_F1')}",
    f"{_cmp('A0_F2')}::{_cmp('A0_F0')}",
]

# Valence codes in the comparison files: 1 supportive, 0 non-directional, -1 undermining.
UNDERMINING = "-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Wilson and article-level bootstrap confidence intervals for "
            "locked gold100 model-performance estimates from existing comparison CSVs."
        )
    )
    parser.add_argument(
        "--matrix-dir",
        action="append",
        default=None,
        help="Matrix directory containing */*_comparison.csv files. Can be repeated.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        help="Named comparison CSV as SYSTEM_ID=path/to/comparison.csv. Can be repeated.",
    )
    parser.add_argument(
        "--paired-comparison",
        action="append",
        default=None,
        help="Paired comparison spec as NAME_A=path::NAME_B=path. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/modeling/performance_cis_gold100_2026-05-27",
        help="Output directory for CI tables and manifest.",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_comparison(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in df.columns:
        if col.startswith("match_") or col == "exact_match_all_fields":
            df[col] = df[col].map(lambda value: str(value).strip().lower() == "true")
    return df


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=path, got: {raw}")
    name, path = raw.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"Invalid named path spec: {raw}")
    return name, Path(path)


def discover_comparisons(matrix_dirs: list[Path], comparison_specs: list[str]) -> dict[str, Path]:
    comparisons: dict[str, Path] = {}
    for matrix_dir in matrix_dirs:
        for path in sorted(matrix_dir.glob("*/*_comparison.csv")):
            system_id = path.parent.name
            if system_id in comparisons:
                raise ValueError(f"Duplicate system_id discovered: {system_id}")
            comparisons[system_id] = path
    for spec in comparison_specs:
        system_id, path = parse_named_path(spec)
        if system_id in comparisons:
            raise ValueError(f"Duplicate system_id supplied: {system_id}")
        comparisons[system_id] = path
    return comparisons


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    # 1.959963984540054 is the 97.5th percentile for the standard normal.
    # The script currently supports the planned 95% interval.
    if not math.isclose(alpha, 0.05):
        raise ValueError("Wilson interval currently supports alpha=0.05 only.")
    z = 1.959963984540054
    p = successes / n
    denom = 1 + (z * z / n)
    centre = p + (z * z / (2 * n))
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    low = max(0.0, (centre - margin) / denom)
    high = min(1.0, (centre + margin) / denom)
    if abs(low) < 1e-12:
        low = 0.0
    if abs(1.0 - high) < 1e-12:
        high = 1.0
    return low, high


def percentile_interval(values: list[float], alpha: float = 0.05) -> tuple[float | None, float | None]:
    clean = sorted(value for value in values if value is not None and not math.isnan(value))
    if not clean:
        return None, None
    lower_pos = (alpha / 2) * (len(clean) - 1)
    upper_pos = (1 - alpha / 2) * (len(clean) - 1)
    return interpolate(clean, lower_pos), interpolate(clean, upper_pos)


def interpolate(values: list[float], position: float) -> float:
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def is_present(value: Any, construct: str) -> bool:
    text = "" if value is None else str(value).strip()
    if not text:
        return False
    return construct in set(part for part in text.split("|") if part)


def safe_divide(num: int | float, den: int | float) -> float | None:
    return num / den if den else None


def binary_counts(gold: list[bool], pred: list[bool]) -> dict[str, int]:
    tp = sum(g and p for g, p in zip(gold, pred))
    tn = sum((not g) and (not p) for g, p in zip(gold, pred))
    fp = sum((not g) and p for g, p in zip(gold, pred))
    fn = sum(g and (not p) for g, p in zip(gold, pred))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": tp + tn + fp + fn}


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denominator = (2 * tp) + fp + fn
    return (2 * tp / denominator) if denominator else 0.0


def binary_metric_values(gold: list[bool], pred: list[bool]) -> dict[str, float | int | None]:
    counts = binary_counts(gold, pred)
    tp = counts["tp"]
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    n = counts["n"]
    return {
        **counts,
        "accuracy": safe_divide(tp + tn, n),
        "precision": safe_divide(tp, tp + fp),
        "sensitivity": safe_divide(tp, tp + fn),
        "specificity": safe_divide(tn, tn + fp),
        "f1": f1_from_counts(tp, fp, fn),
    }


def construct_presence_metrics(df: pd.DataFrame) -> dict[str, float | None]:
    per_construct_f1: list[float] = []
    micro_gold: list[bool] = []
    micro_pred: list[bool] = []
    metrics: dict[str, float | None] = {}
    for construct in CONSTRUCTS:
        gold = [is_present(value, construct) for value in df["gold_c5_presence_d1"]]
        pred = [is_present(value, construct) for value in df["pred_c5_presence_d1"]]
        values = binary_metric_values(gold, pred)
        metrics[f"construct_{construct}_f1"] = values["f1"]
        if values["f1"] is not None:
            per_construct_f1.append(float(values["f1"]))
        micro_gold.extend(gold)
        micro_pred.extend(pred)
    metrics["construct_presence_macro_f1"] = (
        sum(per_construct_f1) / len(per_construct_f1) if per_construct_f1 else None
    )
    metrics["construct_presence_micro_f1"] = binary_metric_values(micro_gold, micro_pred)["f1"]
    return metrics


def scalar_accuracy(df: pd.DataFrame, gold_col: str, pred_col: str) -> float | None:
    if df.empty:
        return None
    return float((df[gold_col].astype(str) == df[pred_col].astype(str)).mean())


def metric_functions() -> dict[str, Callable[[pd.DataFrame], float | None]]:
    functions: dict[str, Callable[[pd.DataFrame], float | None]] = {
        "relevance_accuracy": lambda df: scalar_accuracy(df, "gold_relevance", "pred_relevance"),
        "relevance_sensitivity": lambda df: binary_metric_values(
            [value == "relevant" for value in df["gold_relevance"]],
            [value == "relevant" for value in df["pred_relevance"]],
        )["sensitivity"],
        "relevance_specificity": lambda df: binary_metric_values(
            [value == "relevant" for value in df["gold_relevance"]],
            [value == "relevant" for value in df["pred_relevance"]],
        )["specificity"],
        "relevance_precision": lambda df: binary_metric_values(
            [value == "relevant" for value in df["gold_relevance"]],
            [value == "relevant" for value in df["pred_relevance"]],
        )["precision"],
        "exact_match_all_fields": lambda df: float(df["exact_match_all_fields"].mean()) if not df.empty else None,
        "has_relevant_5c_accuracy_all": lambda df: scalar_accuracy(
            df, "gold_has_relevant_5c", "pred_has_relevant_5c"
        ),
        "has_relevant_5c_accuracy_gold_relevant": lambda df: scalar_accuracy(
            df[df["gold_relevance"] == "relevant"],
            "gold_has_relevant_5c",
            "pred_has_relevant_5c",
        ),
        "disease_accuracy_gold_relevant": lambda df: scalar_accuracy(
            df[df["gold_relevance"] == "relevant"], "gold_disease_1", "pred_disease_1"
        ),
    }
    for construct in CONSTRUCTS:
        functions[f"construct_{construct}_f1"] = (
            lambda df, construct=construct: construct_presence_metrics(df)[f"construct_{construct}_f1"]
        )
    functions["construct_presence_macro_f1"] = lambda df: construct_presence_metrics(df)[
        "construct_presence_macro_f1"
    ]
    functions["construct_presence_micro_f1"] = lambda df: construct_presence_metrics(df)[
        "construct_presence_micro_f1"
    ]
    return functions


def binomial_rows(system_id: str, df: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(metric: str, successes: int, n: int, denominator_note: str) -> None:
        point = safe_divide(successes, n)
        low, high = wilson_interval(successes, n, alpha)
        rows.append(
            {
                "system_id": system_id,
                "metric": metric,
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "method": "wilson",
                "successes": successes,
                "n": n,
                "denominator_note": denominator_note,
            }
        )

    add("relevance_accuracy", int(df["match_relevance"].sum()), len(df), "all locked gold100 articles")
    relevance = binary_metric_values(
        [value == "relevant" for value in df["gold_relevance"]],
        [value == "relevant" for value in df["pred_relevance"]],
    )
    add("relevance_sensitivity", relevance["tp"], relevance["tp"] + relevance["fn"], "gold relevant articles")
    add("relevance_specificity", relevance["tn"], relevance["tn"] + relevance["fp"], "gold not-relevant articles")
    add("relevance_precision", relevance["tp"], relevance["tp"] + relevance["fp"], "predicted relevant articles")
    add("exact_match_all_fields", int(df["exact_match_all_fields"].sum()), len(df), "all locked gold100 articles")
    add(
        "has_relevant_5c_accuracy_all",
        int(df["match_has_relevant_5c"].sum()),
        len(df),
        "all locked gold100 articles; retained for comparability with matrix overview",
    )
    relevant_rows = df[df["gold_relevance"] == "relevant"]
    add(
        "has_relevant_5c_accuracy_gold_relevant",
        int(relevant_rows["match_has_relevant_5c"].sum()),
        len(relevant_rows),
        "gold relevant articles",
    )
    add(
        "disease_accuracy_gold_relevant",
        int(relevant_rows["match_disease_1"].sum()),
        len(relevant_rows),
        "gold relevant articles",
    )
    for construct, field in VALENCE_FIELDS.items():
        construct_rows = df[[is_present(value, construct) for value in df["gold_c5_presence_d1"]]]
        add(
            f"valence_{construct}_accuracy_when_gold_present",
            int(construct_rows[f"match_{field}"].sum()),
            len(construct_rows),
            f"gold-present {construct} articles",
        )
    return rows


def bootstrap_rows(
    system_id: str,
    df: pd.DataFrame,
    functions: dict[str, Callable[[pd.DataFrame], float | None]],
    reps: int,
    alpha: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(df)
    target_metrics = [
        *(f"construct_{construct}_f1" for construct in CONSTRUCTS),
        *(f"construct_{construct}_f1_zero_undefined" for construct in CONSTRUCTS),
        "construct_presence_macro_f1",
        "construct_presence_macro_f1_zero_undefined",
        "construct_presence_macro_f1_excluding_calculation",
        "construct_presence_micro_f1",
    ]
    construct_gold = np.array(
        [[is_present(value, construct) for construct in CONSTRUCTS] for value in df["gold_c5_presence_d1"]],
        dtype=bool,
    )
    construct_pred = np.array(
        [[is_present(value, construct) for construct in CONSTRUCTS] for value in df["pred_c5_presence_d1"]],
        dtype=bool,
    )
    seed = _stable_seed(base_seed, system_id)
    np_rng = np.random.default_rng(seed)
    distributions = {metric: np.empty(reps, dtype=float) for metric in target_metrics}
    for _ in range(reps):
        sample_indices = np_rng.integers(0, n, size=n)
        values = fast_construct_presence_metrics(construct_gold[sample_indices], construct_pred[sample_indices])
        for metric in target_metrics:
            value = values[metric]
            distributions[metric][_] = np.nan if value is None else float(value)
    for metric in target_metrics:
        point = fast_construct_presence_metrics(construct_gold, construct_pred)[metric]
        low, high = percentile_interval(distributions[metric].tolist(), alpha)
        rows.append(
            {
                "system_id": system_id,
                "metric": metric,
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "method": "article_level_percentile_bootstrap",
                "bootstrap_reps": reps,
                "n": n,
                "denominator_note": "resampled locked articles",
            }
        )
    return rows


def fast_construct_presence_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    tp = np.logical_and(gold, pred).sum(axis=0)
    fp = np.logical_and(~gold, pred).sum(axis=0)
    fn = np.logical_and(gold, ~pred).sum(axis=0)
    f1_values: list[float | None] = []
    metrics: dict[str, float | None] = {}
    for idx, construct in enumerate(CONSTRUCTS):
        f1 = f1_from_counts(int(tp[idx]), int(fp[idx]), int(fn[idx]))
        metrics[f"construct_{construct}_f1"] = f1
        metrics[f"construct_{construct}_f1_zero_undefined"] = 0.0 if f1 is None else f1
        f1_values.append(f1)
    clean_f1 = [value for value in f1_values if value is not None]
    metrics["construct_presence_macro_f1"] = sum(clean_f1) / len(clean_f1) if clean_f1 else None
    metrics["construct_presence_macro_f1_zero_undefined"] = sum(
        0.0 if value is None else value for value in f1_values
    ) / len(f1_values)
    # Mean F1 over the four constructs other than the exploratory Calculation construct.
    four = [value for construct, value in zip(CONSTRUCTS, f1_values) if construct != "calculation"]
    metrics["construct_presence_macro_f1_excluding_calculation"] = (
        sum(four) / len(four) if all(value is not None for value in four) else None
    )
    metrics["construct_presence_micro_f1"] = f1_from_counts(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    return metrics


def fast_metric_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "doc_id": df["doc_id"].astype(str).to_numpy(),
        "relevance_match": df["match_relevance"].to_numpy(dtype=bool),
        "exact_match_all_fields": df["exact_match_all_fields"].to_numpy(dtype=bool),
        "has_5c_match": df["match_has_relevant_5c"].to_numpy(dtype=bool),
        "disease_match": df["match_disease_1"].to_numpy(dtype=bool),
        "gold_relevant": (df["gold_relevance"].astype(str) == "relevant").to_numpy(dtype=bool),
        "construct_gold": np.array(
            [[is_present(value, construct) for construct in CONSTRUCTS] for value in df["gold_c5_presence_d1"]],
            dtype=bool,
        ),
        "construct_pred": np.array(
            [[is_present(value, construct) for construct in CONSTRUCTS] for value in df["pred_c5_presence_d1"]],
            dtype=bool,
        ),
    }


def fast_scalar_metrics(arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, float | None]:
    if indices is None:
        rel_match = arrays["relevance_match"]
        exact = arrays["exact_match_all_fields"]
        has_5c = arrays["has_5c_match"]
        disease = arrays["disease_match"]
        gold_relevant = arrays["gold_relevant"]
        construct_gold = arrays["construct_gold"]
        construct_pred = arrays["construct_pred"]
    else:
        rel_match = arrays["relevance_match"][indices]
        exact = arrays["exact_match_all_fields"][indices]
        has_5c = arrays["has_5c_match"][indices]
        disease = arrays["disease_match"][indices]
        gold_relevant = arrays["gold_relevant"][indices]
        construct_gold = arrays["construct_gold"][indices]
        construct_pred = arrays["construct_pred"][indices]
    out: dict[str, float | None] = {
        "relevance_accuracy": float(rel_match.mean()) if rel_match.size else None,
        "exact_match_all_fields": float(exact.mean()) if exact.size else None,
        "has_relevant_5c_accuracy_all": float(has_5c.mean()) if has_5c.size else None,
        "has_relevant_5c_accuracy_gold_relevant": (
            float(has_5c[gold_relevant].mean()) if gold_relevant.sum() else None
        ),
        "disease_accuracy_gold_relevant": float(disease[gold_relevant].mean()) if gold_relevant.sum() else None,
    }
    out.update(fast_construct_presence_metrics(construct_gold, construct_pred))
    return out


def parse_paired_spec(raw: str) -> tuple[str, Path, str, Path]:
    if "::" not in raw:
        raise ValueError(f"Expected NAME_A=path::NAME_B=path, got: {raw}")
    left, right = raw.split("::", 1)
    left_name, left_path = parse_named_path(left)
    right_name, right_path = parse_named_path(right)
    return left_name, left_path, right_name, right_path


def paired_bootstrap_rows(
    specs: list[str],
    functions: dict[str, Callable[[pd.DataFrame], float | None]],
    reps: int,
    alpha: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = [
        "relevance_accuracy",
        "exact_match_all_fields",
        "has_relevant_5c_accuracy_all",
        "has_relevant_5c_accuracy_gold_relevant",
        "disease_accuracy_gold_relevant",
        "construct_presence_macro_f1",
        "construct_presence_macro_f1_zero_undefined",
        "construct_presence_micro_f1",
        *(f"construct_{construct}_f1" for construct in CONSTRUCTS),
        *(f"construct_{construct}_f1_zero_undefined" for construct in CONSTRUCTS),
    ]
    for spec in specs:
        left_name, left_path, right_name, right_path = parse_paired_spec(spec)
        left = read_comparison(left_path).sort_values("doc_id").reset_index(drop=True)
        right = read_comparison(right_path).sort_values("doc_id").reset_index(drop=True)
        if left["doc_id"].tolist() != right["doc_id"].tolist():
            raise ValueError(f"Doc IDs do not align for paired comparison {left_name} vs {right_name}.")
        n = len(left)
        left_arrays = fast_metric_arrays(left)
        right_arrays = fast_metric_arrays(right)
        seed = _stable_seed(base_seed, f"{left_name}_minus_{right_name}")
        np_rng = np.random.default_rng(seed)
        distributions = {metric: np.empty(reps, dtype=float) for metric in metrics}
        for _ in range(reps):
            sample_indices = np_rng.integers(0, n, size=n)
            left_values = fast_scalar_metrics(left_arrays, sample_indices)
            right_values = fast_scalar_metrics(right_arrays, sample_indices)
            for metric in metrics:
                left_value = left_values[metric]
                right_value = right_values[metric]
                if left_value is not None and right_value is not None:
                    distributions[metric][_] = float(left_value) - float(right_value)
                else:
                    distributions[metric][_] = np.nan
        left_point_values = fast_scalar_metrics(left_arrays)
        right_point_values = fast_scalar_metrics(right_arrays)
        for metric in metrics:
            left_point = left_point_values[metric]
            right_point = right_point_values[metric]
            point = None if left_point is None or right_point is None else float(left_point) - float(right_point)
            low, high = percentile_interval(distributions[metric].tolist(), alpha)
            rows.append(
                {
                    "comparison": f"{left_name}_minus_{right_name}",
                    "left_system_id": left_name,
                    "right_system_id": right_name,
                    "metric": metric,
                    "point_difference": point,
                    "ci_low": low,
                    "ci_high": high,
                    "method": "paired_article_level_percentile_bootstrap",
                    "bootstrap_reps": reps,
                    "n": n,
                }
            )
    return rows


def valence_rows(
    system_id: str,
    df: pd.DataFrame,
    reps: int,
    alpha: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Valence accuracy against the majority-class baseline and undermining-valence detection.

    Valence accuracy: among articles in which the reference marks the construct present, the
    proportion whose three-class valence was predicted correctly (an article the pipeline did not
    code for the construct counts as wrong); Wilson interval. Majority-class baseline: the accuracy
    of always predicting the most frequent reference valence among those articles.

    Undermining detection (valence code -1), per construct: recall = correct / reference-undermining,
    precision = correct / predicted-undermining, where correct means both reference and prediction
    code the construct as undermining in the same article; Wilson intervals. Scored end to end, so
    an article dropped by the relevance classifier has no predicted valence.

    Pooled over the five constructs: the same ratios of summed counts. Because one article can hold
    several undermining labels, the pooled interval is an article-level percentile bootstrap
    (articles resampled with replacement, pooled counts recomputed; seed from _stable_seed).
    """
    rows: list[dict[str, Any]] = []
    gold_under = np.zeros((len(df), len(CONSTRUCTS)), dtype=bool)
    pred_under = np.zeros((len(df), len(CONSTRUCTS)), dtype=bool)

    def add(metric: str, construct: str, successes: int, n: int, method: str, note: str,
            low: float | None = None, high: float | None = None) -> None:
        if method == "wilson":
            low, high = wilson_interval(successes, n, alpha)
        rows.append(
            {
                "system_id": system_id,
                "metric": metric,
                "construct": construct,
                "point_estimate": safe_divide(successes, n),
                "ci_low": low,
                "ci_high": high,
                "method": method,
                "successes": successes,
                "n": n,
                "note": note,
            }
        )

    for idx, (construct, field) in enumerate(VALENCE_FIELDS.items()):
        gold_present = df["gold_c5_presence_d1"].map(lambda value, c=construct: is_present(value, c))
        present = df[gold_present]
        add(
            "valence_accuracy_when_gold_present", construct,
            int(present[f"match_{field}"].sum()), len(present), "wilson",
            "reference-present articles; three-class valence",
        )
        counts = present[f"gold_{field}"].astype(str).value_counts()
        if len(present):
            majority_label = str(counts.index[0])
            add(
                "valence_majority_class_baseline", construct,
                int(counts.iloc[0]), len(present), "fixed_rule",
                f"always predict the most frequent reference valence ({majority_label}); "
                "distribution " + "/".join(f"{label}:{int(counts.get(label, 0))}" for label in ["1", "0", "-1"]),
            )
        gold_under[:, idx] = (df[f"gold_{field}"].astype(str) == UNDERMINING).to_numpy()
        pred_under[:, idx] = (df[f"pred_{field}"].astype(str) == UNDERMINING).to_numpy()
        correct = int((gold_under[:, idx] & pred_under[:, idx]).sum())
        add("undermining_recall", construct, correct, int(gold_under[:, idx].sum()), "wilson",
            "correct / reference-undermining")
        add("undermining_precision", construct, correct, int(pred_under[:, idx].sum()), "wilson",
            "correct / predicted-undermining")

    per_article_correct = (gold_under & pred_under).sum(axis=1)
    per_article_gold = gold_under.sum(axis=1)
    per_article_pred = pred_under.sum(axis=1)
    rng = np.random.default_rng(_stable_seed(base_seed, f"{system_id}:undermining_pooled"))
    n = len(df)
    recall_draws = np.empty(reps, dtype=float)
    precision_draws = np.empty(reps, dtype=float)
    for rep in range(reps):
        sample = rng.integers(0, n, size=n)
        c = per_article_correct[sample].sum()
        g = per_article_gold[sample].sum()
        p = per_article_pred[sample].sum()
        recall_draws[rep] = c / g if g else np.nan
        precision_draws[rep] = c / p if p else np.nan
    correct_total = int(per_article_correct.sum())
    for metric, denominator, draws, note in [
        ("undermining_recall", int(per_article_gold.sum()), recall_draws, "correct / reference-undermining"),
        ("undermining_precision", int(per_article_pred.sum()), precision_draws, "correct / predicted-undermining"),
    ]:
        low, high = percentile_interval(draws.tolist(), alpha)
        add(metric, "pooled", correct_total, denominator, "article_level_percentile_bootstrap",
            f"{note}; summed over the five constructs; {reps} article resamples",
            low, high)
    return rows


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_dirs = [Path(path) for path in (args.matrix_dir or DEFAULT_MATRIX_DIRS)]
    comparison_specs = args.comparison or []
    paired_specs = args.paired_comparison or DEFAULT_PAIRED_COMPARISONS
    comparisons = discover_comparisons(matrix_dirs, comparison_specs)

    functions = metric_functions()

    binomial: list[dict[str, Any]] = []
    bootstrap: list[dict[str, Any]] = []
    valence: list[dict[str, Any]] = []
    for system_id, path in sorted(comparisons.items()):
        df = read_comparison(path)
        binomial.extend(binomial_rows(system_id, df, args.alpha))
        bootstrap.extend(bootstrap_rows(system_id, df, functions, args.bootstrap_reps, args.alpha, args.seed))
        valence.extend(valence_rows(system_id, df, args.bootstrap_reps, args.alpha, args.seed))

    paired = paired_bootstrap_rows(paired_specs, functions, args.bootstrap_reps, args.alpha, args.seed)

    binomial_path = output_dir / "gold100_binomial_metric_wilson_cis.csv"
    bootstrap_path = output_dir / "gold100_f1_metric_bootstrap_cis.csv"
    paired_path = output_dir / "gold100_paired_metric_differences_bootstrap_cis.csv"
    valence_path = output_dir / "gold100_valence_metric_cis.csv"
    manifest_path = output_dir / "performance_ci_manifest.json"

    pd.DataFrame(binomial).to_csv(binomial_path, index=False)
    pd.DataFrame(bootstrap).to_csv(bootstrap_path, index=False)
    pd.DataFrame(paired).to_csv(paired_path, index=False)
    pd.DataFrame(valence).to_csv(valence_path, index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "locked_gold100_performance_confidence_intervals",
        "alpha": args.alpha,
        "confidence_level": 1 - args.alpha,
        "bootstrap_reps": args.bootstrap_reps,
        "seed": args.seed,
        "matrix_dirs": [str(path) for path in matrix_dirs],
        "comparison_files": {system_id: str(path) for system_id, path in sorted(comparisons.items())},
        "paired_comparisons": paired_specs,
        "outputs": {
            "binomial_wilson_cis": str(binomial_path),
            "f1_bootstrap_cis": str(bootstrap_path),
            "paired_bootstrap_cis": str(paired_path),
            "valence_cis": str(valence_path),
        },
        "methods_note": (
            "Wilson score intervals are used for binomial metrics. Non-parametric "
            "article-level percentile bootstrap intervals are used for F1 metrics and for the "
            "pooled undermining-valence recall and precision. "
            "Paired article-level bootstrap intervals are used for same-article model comparisons."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Saved Wilson CIs: {binomial_path}")
    print(f"Saved F1 bootstrap CIs: {bootstrap_path}")
    print(f"Saved paired bootstrap CIs: {paired_path}")
    print(f"Saved valence CIs: {valence_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
