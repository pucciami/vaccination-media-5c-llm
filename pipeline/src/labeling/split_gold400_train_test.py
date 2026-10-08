import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = Path("data/processed/gold400_tidy.csv")
EXPORT_PATH = Path("annotations/backups/exports/project_1_gold400_double_annotated_v2_cleaned.json")
ASSIGNMENT_PATH = Path("data/processed/gold400_split_assignment.csv")
TRAIN_PATH = Path("data/processed/gold300_train.csv")
TEST_PATH = Path("data/processed/gold100_test.csv")


def get_args():
    parser = argparse.ArgumentParser(
        description="Create a reproducible 300/100 split for the 400-document gold dataset."
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument("--export", default=str(EXPORT_PATH))
    parser.add_argument("--assignment-output", default=str(ASSIGNMENT_PATH))
    parser.add_argument("--train-output", default=str(TRAIN_PATH))
    parser.add_argument("--test-output", default=str(TEST_PATH))
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_year_map(export_path):
    with open(export_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for task in data:
        rows.append({
            "labelstudio_task_id": task.get("id"),
            "source_doc_id": task.get("data", {}).get("doc_id"),
            "year": task.get("data", {}).get("year"),
            "published_at": task.get("data", {}).get("published_at"),
            "outlet": task.get("data", {}).get("outlet"),
            "title": task.get("data", {}).get("title"),
        })
    return pd.DataFrame(rows)


def provisional_disease(row):
    ann1 = row.get("ann1_disease_1")
    ann2 = row.get("ann2_disease_1")

    if pd.notna(ann1) and pd.notna(ann2) and ann1 == ann2:
        return ann1
    if pd.notna(ann1):
        return ann1
    if pd.notna(ann2):
        return ann2
    return "none"


def disease_group(value):
    if pd.isna(value):
        return "none"
    text = str(value)
    if text in {"covid19", "influenza", "hpv", "measles", "other"}:
        return text
    if text in {"unspecified", "none"}:
        return "rare_or_none"
    return "rare_or_none"


def year_bucket(value):
    if pd.isna(value):
        return "year_unknown"
    year = int(value)
    if year <= 2015:
        return "2010_2015"
    if year <= 2019:
        return "2016_2019"
    if year == 2020:
        return "2020"
    if year == 2021:
        return "2021"
    if year == 2022:
        return "2022"
    return "2023_2024"


def safe_strata(df):
    combined = df["disease_group"] + "__" + df["year_bucket"]
    counts = combined.value_counts()
    if (counts >= 2).all():
        return combined, "disease_group + year_bucket"

    disease_only = df["disease_group"]
    counts = disease_only.value_counts()
    if (counts >= 2).all():
        return disease_only, "disease_group"

    year_only = df["year_bucket"]
    counts = year_only.value_counts()
    if (counts >= 2).all():
        return year_only, "year_bucket"

    return None, "random_only"


def main():
    args = get_args()

    df = pd.read_csv(args.dataset)
    meta = load_year_map(args.export)

    merged = df.merge(
        meta,
        on=["labelstudio_task_id", "source_doc_id"],
        how="left",
        validate="one_to_one",
    )

    merged["provisional_disease"] = merged.apply(provisional_disease, axis=1)
    merged["disease_group"] = merged["provisional_disease"].apply(disease_group)
    merged["year_bucket"] = merged["year"].apply(year_bucket)

    strata, strategy = safe_strata(merged)

    if strata is None:
        train_idx, test_idx = train_test_split(
            merged.index,
            test_size=args.test_size,
            random_state=args.random_seed,
            shuffle=True,
        )
    else:
        train_idx, test_idx = train_test_split(
            merged.index,
            test_size=args.test_size,
            random_state=args.random_seed,
            shuffle=True,
            stratify=strata,
        )

    merged["split"] = "train"
    merged.loc[test_idx, "split"] = "test"

    assignment_cols = [
        "doc_id",
        "labelstudio_task_id",
        "source_doc_id",
        "year",
        "published_at",
        "outlet",
        "title",
        "provisional_disease",
        "disease_group",
        "year_bucket",
        "split",
    ]
    assignment = merged[assignment_cols].copy()

    train_df = merged[merged["split"] == "train"].copy()
    test_df = merged[merged["split"] == "test"].copy()

    Path(args.assignment_output).parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(args.assignment_output, index=False)
    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)

    print(f"Split strategy: {strategy}")
    print(f"Random seed: {args.random_seed}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"\nSaved assignments: {args.assignment_output}")
    print(f"Saved train split: {args.train_output}")
    print(f"Saved test split: {args.test_output}")

    print("\nTrain disease groups:")
    print(train_df["disease_group"].value_counts())
    print("\nTest disease groups:")
    print(test_df["disease_group"].value_counts())

    print("\nTrain year buckets:")
    print(train_df["year_bucket"].value_counts())
    print("\nTest year buckets:")
    print(test_df["year_bucket"].value_counts())


if __name__ == "__main__":
    main()
