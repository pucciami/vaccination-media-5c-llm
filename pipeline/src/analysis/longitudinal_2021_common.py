from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


CONSTRUCTS = [
    "confidence",
    "complacency",
    "constraints",
    "calculation",
    "collective_responsibility",
]

CONSTRUCT_LABELS = {
    "confidence": "Confidence",
    "complacency": "Complacency",
    "constraints": "Constraints",
    "calculation": "Calculation",
    "collective_responsibility": "Collective responsibility",
}

VALENCE_FIELDS = {
    "confidence": "valence_confidence_d1",
    "complacency": "valence_complacency_d1",
    "constraints": "valence_constraints_d1",
    "calculation": "valence_calculation_d1",
    "collective_responsibility": "valence_collective_d1",
}

VALENCE_LABELS = {
    -1: "undermining",
    0: "neutral",
    1: "supportive",
}

COVID_REGEX = r"covid|koronavirus|sars[- ]?cov[- ]?2"
VACCINATION_REGEX = r"cepiv|cepljen|cepljiv|cepljenj|precep|vakcin|imuniz|vaccin"
JANSSEN_REGEX = r"janssen|johnson\s*&\s*johnson|j\s*&\s*j"
DEFAULT_LEXICAL_SEARCH_CHARS = 2100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL row {line_no} in {path}") from exc
    return rows


def normalize_doc_id(value: Any) -> str:
    return str(value).strip()


def normalize_relevance(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    if text == "relevant":
        return "relevant"
    if text in {"not relevant", "notrelevant", "irrelevant"}:
        return "not_relevant"
    return None


def normalize_presence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or text in {"null", "None", "nan"}:
            return []
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                raw = ast.literal_eval(text)
        elif "|" in text:
            raw = text.split("|")
        else:
            raw = [text]
    else:
        raw = [value]

    normalized = []
    for item in raw:
        label = str(item).strip().lower()
        if label in CONSTRUCTS and label not in normalized:
            normalized.append(label)
    return sorted(normalized, key=CONSTRUCTS.index)


def normalize_valence(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text in {"null", "none", "nan"}:
        return None
    mapping = {
        "-1": -1,
        "-1.0": -1,
        "undermining": -1,
        "negative": -1,
        "0": 0,
        "0.0": 0,
        "neutral": 0,
        "mixed": 0,
        "descriptive": 0,
        "1": 1,
        "+1": 1,
        "1.0": 1,
        "+1.0": 1,
        "supportive": 1,
        "positive": 1,
    }
    return mapping.get(text)


def add_time_fields(df: pd.DataFrame, date_col: str = "published_at") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["date"] = out[date_col].dt.date.astype("string")
    out["year"] = out[date_col].dt.year.astype("Int64")
    out["month"] = out[date_col].dt.to_period("M").astype("string")
    out["week_start"] = (out[date_col].dt.normalize() - pd.to_timedelta(out[date_col].dt.weekday, unit="D")).dt.date.astype("string")
    out["iso_year"] = out[date_col].dt.isocalendar().year.astype("Int64")
    out["iso_week"] = out[date_col].dt.isocalendar().week.astype("Int64")
    return out


def fortnight_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    week_start = dates.dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit="D")
    origin = pd.Timestamp("2021-01-04")
    fortnight_index = ((week_start - origin).dt.days // 14).clip(lower=0)
    return (origin + pd.to_timedelta(fortnight_index * 14, unit="D")).dt.date.astype("string")


def lexical_flags(title: pd.Series, model_text: pd.Series, max_chars: int | None = DEFAULT_LEXICAL_SEARCH_CHARS) -> pd.DataFrame:
    body = model_text.fillna("").astype(str)
    if max_chars and max_chars > 0:
        body = body.str.slice(0, max_chars)
    haystack = (title.fillna("").astype(str) + "\n" + body).str.lower()
    return pd.DataFrame(
        {
            "covid_lexical": haystack.str.contains(COVID_REGEX, flags=re.IGNORECASE, regex=True, na=False),
            "vaccination_lexical": haystack.str.contains(VACCINATION_REGEX, flags=re.IGNORECASE, regex=True, na=False),
            "janssen_lexical": haystack.str.contains(JANSSEN_REGEX, flags=re.IGNORECASE, regex=True, na=False),
        }
    )


def summarize_counts(df: pd.DataFrame) -> dict[str, Any]:
    dates = pd.to_datetime(df["published_at"], errors="coerce")
    return {
        "n_rows": int(len(df)),
        "date_min": None if dates.dropna().empty else dates.min().date().isoformat(),
        "date_max": None if dates.dropna().empty else dates.max().date().isoformat(),
        "n_outlets": int(df["outlet"].fillna("").replace("", pd.NA).dropna().nunique()) if "outlet" in df else None,
    }
