from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DOCS_PATH = Path("data/interim/documents_media.parquet")
OUT_JSON = Path("data/processed/ls_articles_gold_300.json")

N_TOTAL = 300
SEED = 42

# Quotas per disease proxy bucket
QUOTA = {
    "measles": 40,
    "covid19": 70,
    "influenza": 60,
    "hpv": 40,
    "vax_unspecified": 60,
    "other": 30,
}

# Soft cap: max articles per outlet WITHIN each bucket
# (prevents one outlet dominating any disease bucket)
OUTLET_CAP_PER_BUCKET = 15

PATTERNS = {
    "measles": re.compile(r"\b(ošpice|ospice|measles)\b", re.IGNORECASE),
    "covid19": re.compile(r"\b(covid|koronavirus|sars\-?cov\-?2)\b", re.IGNORECASE),
    "influenza": re.compile(r"\b(gripa|influenza)\b", re.IGNORECASE),
    "hpv": re.compile(r"\b(hpv|papilom)\b", re.IGNORECASE),
}

def bucket_row(topics: str, title: str, text: str) -> str:
    s = f"{topics}\n{title}\n{text[:800]}"
    for k, pat in PATTERNS.items():
        if pat.search(s):
            return k
    if re.search(r"\b(cepljen|cepljenj|cepiv)\b", s, re.IGNORECASE):
        return "vax_unspecified"
    return "other"

def sample_bucket(sub: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Sample n rows from sub with:
    - approximate year diversity
    - outlet cap within the bucket
    """
    rng = random.Random(seed)

    # Remove unusable rows just in case
    sub = sub.copy()
    sub = sub[sub["text"].astype(str).str.len() > 0]
    if len(sub) == 0:
        return sub

    # Build year groups
    years = [y for y in sub["year"].dropna().unique().tolist() if int(y) > 0]
    years = sorted(years)
    if not years:
        years = [0]

    # Target per-year picks (approx)
    per_year = max(1, n // max(1, len(years)))

    picked_idx = []
    outlet_counts = Counter()

    # Helper: pick up to k rows from a frame respecting outlet cap
    def pick_from_frame(frame: pd.DataFrame, k: int) -> list[int]:
        if k <= 0 or len(frame) == 0:
            return []
        # Shuffle indices deterministically
        idxs = frame.index.tolist()
        rng.shuffle(idxs)
        out = []
        for ix in idxs:
            outlet = str(frame.at[ix, "outlet"])
            if outlet_counts[outlet] < OUTLET_CAP_PER_BUCKET:
                out.append(ix)
                outlet_counts[outlet] += 1
                if len(out) >= k:
                    break
        return out

    # Pass 1: take per_year from each year
    for y in years:
        frame = sub[sub["year"] == y] if y != 0 else sub
        picked_idx.extend(pick_from_frame(frame, per_year))

    # Pass 2: top up to n from remaining pool, still respecting cap
    if len(picked_idx) < n:
        remaining = sub.drop(index=picked_idx, errors="ignore")
        need = n - len(picked_idx)
        picked_idx.extend(pick_from_frame(remaining, need))

    # Pass 3: if cap is too strict and we still lack items, relax cap gently
    # (rare; ensures quotas are met)
    relax_cap = OUTLET_CAP_PER_BUCKET
    while len(picked_idx) < n and relax_cap < 1000:
        relax_cap += 5
        remaining = sub.drop(index=picked_idx, errors="ignore")
        idxs = remaining.index.tolist()
        rng.shuffle(idxs)
        for ix in idxs:
            outlet = str(remaining.at[ix, "outlet"])
            if outlet_counts[outlet] < relax_cap:
                picked_idx.append(ix)
                outlet_counts[outlet] += 1
                if len(picked_idx) >= n:
                    break

    picked = sub.loc[picked_idx].copy()
    # If we somehow overshot, downsample (shouldn't happen)
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed)

    return picked

def main():
    assert sum(QUOTA.values()) == N_TOTAL
    random.seed(SEED)

    df = pd.read_parquet(DOCS_PATH)

    dt = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.assign(year=dt.dt.year.fillna(0).astype(int))

    df["bucket"] = df.apply(
        lambda r: bucket_row(str(r["topics"]), str(r["title"]), str(r["text"])),
        axis=1
    )

    samples = []
    for b, n in QUOTA.items():
        sub = df[df["bucket"] == b].copy()
        if len(sub) == 0:
            print(f"WARNING: bucket '{b}' has 0 candidates")
            continue
        picked = sample_bucket(sub, n=n, seed=SEED + hash(b) % 10_000)
        if len(picked) < n:
            print(f"WARNING: bucket '{b}' only sampled {len(picked)}/{n} (not enough candidates?)")
        samples.append(picked)

    samp = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=SEED)

    # Label Studio tasks
    tasks = []
    for _, r in samp.iterrows():
        tasks.append({
            "data": {
                "doc_id": r["doc_id"],
                "published_at": r["published_at"],
                "outlet": r["outlet"],
                "title": r["title"],
                "topics": r["topics"],
                "text": r["text"],
                "bucket_hint": r["bucket"],
                "year": int(r["year"]),
            }
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote: {OUT_JSON} tasks={len(tasks)}")
    print("\nBucket counts:")
    print(samp["bucket"].value_counts())

    print("\nTop outlets overall (sample):")
    print(samp["outlet"].value_counts().head(15))

    print("\nYear coverage (sample):")
    print(samp["year"].value_counts().sort_index().head(20))
    print("... last years:")
    print(samp["year"].value_counts().sort_index().tail(10))

if __name__ == "__main__":
    main()