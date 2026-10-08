"""Build examples/synthetic_articles.csv from the synthetic articles below.

The six articles are invented for this package. They are not drawn from the
restricted corpus and describe no real event, outlet or person. They exist so
that the input format and the text preparation step can be exercised without
access to the corpus. They are not a test set and carry no reference labels.

Run from the package root:

    python examples/build_example_input.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "pipeline"))

from src.common.media_text import build_model_text  # noqa: E402

ARTICLES = [
    {
        "doc_id": "syn-001",
        "published_at": "2021-10-01",
        "title": "Strokovnjaki: korist cepljenja proti covidu-19 presega tveganja",
        "text": (
            "Strokovna skupina je po pregledu podatkov ocenila, da so resni neželeni učinki "
            "po cepljenju zelo redki, korist cepljenja pa jih jasno presega. Poudarili so, da "
            "sistem spremljanja varnosti cepiv deluje in da so vsa poročila skrbno pregledana."
        ),
    },
    {
        "doc_id": "syn-002",
        "published_at": "2021-10-02",
        "title": "Po zapletu več vprašanj o varnosti cepiva",
        "text": (
            "Po poročilu o hudem zapletu po cepljenju se med ljudmi širi dvom o varnosti cepiva. "
            "Nekateri sogovorniki so dejali, da pristojnim ne zaupajo več in da cepljenje odlagajo, "
            "dokler ne bo znanih več podatkov."
        ),
    },
    {
        "doc_id": "syn-003",
        "published_at": "2019-11-12",
        "title": "Gripa letos še ni razširjena, zanimanje za cepljenje majhno",
        "text": (
            "V ambulantah opažajo, da se mnogi za cepljenje proti gripi ne odločijo, ker menijo, "
            "da je gripa blaga bolezen in jih ne ogroža. Zdravniki opozarjajo, da lahko bolezen "
            "pri starejših poteka težje."
        ),
    },
    {
        "doc_id": "syn-004",
        "published_at": "2021-03-18",
        "title": "Dolge vrste pred cepilnim centrom",
        "text": (
            "Pred cepilnim centrom so ljudje čakali več ur. Prebivalci oddaljenih krajev so "
            "povedali, da do cepilnega mesta nimajo prevoza, termini pa so razpisani le v "
            "dopoldanskem času, ko so v službi."
        ),
    },
    {
        "doc_id": "syn-005",
        "published_at": "2017-04-05",
        "title": "Ošpice: s cepljenjem ščitimo tudi tiste, ki se ne morejo cepiti",
        "text": (
            "Pediatrinja je pojasnila, da visoka precepljenost varuje tudi dojenčke in otroke z "
            "oslabljenim imunskim sistemom, ki cepiva ne morejo prejeti. Cepljenje je po njenih "
            "besedah dejanje solidarnosti do skupnosti."
        ),
    },
    {
        "doc_id": "syn-006",
        "published_at": "2018-06-20",
        "title": "Občina obnavlja cesto skozi središče mesta",
        "text": (
            "Občina je začela obnovo glavne ceste. Dela bodo trajala do jeseni, promet bo v tem "
            "času potekal izmenično enosmerno. Med deli bo zaprt tudi parkirni prostor pred "
            "zdravstvenim domom."
        ),
    },
]


def main() -> None:
    out_path = PACKAGE_ROOT / "examples" / "synthetic_articles.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["doc_id", "published_at", "title", "text", "model_text"],
            lineterminator="\n",
        )
        writer.writeheader()
        for article in ARTICLES:
            row = dict(article)
            row["model_text"] = build_model_text(article["title"], article["text"])
            writer.writerow(row)
    print(f"Wrote {out_path} ({len(ARTICLES)} synthetic articles)")


if __name__ == "__main__":
    main()
