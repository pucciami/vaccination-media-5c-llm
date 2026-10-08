#!/usr/bin/env python3
"""Runs INSIDE an HF Job: translate the gold100 Slovenian source to English with
NLLB-200 and upload the model-input CSV to a private dataset repo.

Engine: NLLB-200 (Meta; Costa-jussa et al. 2022). Deterministic beam search,
decoupled from the Qwen classifier under evaluation. Output `model_text`/`text`
columns hold the English translation so the file drops into the stage1 gate and
stage2 extractor jobs unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SRC_LANG = "slv_Latn"
TGT_LANG = "eng_Latn"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def chunk_text(text: str, max_chars: int = 480) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for sent in sentences:
        if len(sent) > max_chars:
            if cur:
                chunks.append(cur); cur = ""
            buf = ""
            for w in sent.split(" "):
                if len(buf) + len(w) + 1 > max_chars:
                    chunks.append(buf); buf = w
                else:
                    buf = (buf + " " + w).strip()
            if buf:
                cur = buf
            continue
        if len(cur) + len(sent) + 1 > max_chars:
            chunks.append(cur); cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        chunks.append(cur)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-dir", default="/job-output/nllb_translation")
    ap.add_argument("--private-output-repo", required=True)
    ap.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-chars", type=int, default=480)
    ap.add_argument("--text-column", default="text_sl")
    ap.add_argument("--id-column", default="doc_id")
    ap.add_argument("--token-env", default="HF_TOKEN")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from huggingface_hub import HfApi

    token = os.environ.get(args.token_env)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input_csv, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision, torch_dtype=dtype).to(device)
    model.eval()
    bos = tok.convert_tokens_to_ids(TGT_LANG)

    def translate(text: str) -> str:
        outs = []
        for chunk in chunk_text(text, args.max_chars):
            enc = tok(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, forced_bos_token_id=bos, num_beams=args.num_beams,
                                     do_sample=False, max_new_tokens=args.max_new_tokens)
            outs.append(tok.batch_decode(gen, skip_special_tokens=True)[0].strip())
        return " ".join(outs).strip()

    out_rows = []
    for i, r in enumerate(rows, start=1):
        doc_id = (r.get(args.id_column) or "").strip()
        src = r.get(args.text_column) or ""
        en = translate(src)
        out_rows.append({
            "doc_id": doc_id,
            "labelstudio_task_id": r.get("labelstudio_task_id", doc_id),
            "source_doc_id": (r.get("source_doc_id") or "").strip(),
            "src_lang": SRC_LANG, "tgt_lang": TGT_LANG,
            "title": "", "text": en, "model_text": en, "source_text_sl": src,
        })
        print(f"[{i}/{len(rows)}] {doc_id} {len(src)}->{len(en)}", flush=True)

    out_csv = out_dir / "english_gold100_translated_model_input.csv"
    cols = ["doc_id", "labelstudio_task_id", "source_doc_id", "src_lang", "tgt_lang",
            "title", "text", "model_text", "source_text_sl"]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out_rows)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": {"model": args.model, "revision": args.revision, "src_lang": SRC_LANG,
                   "tgt_lang": TGT_LANG, "num_beams": args.num_beams, "do_sample": False,
                   "max_new_tokens": args.max_new_tokens, "max_chars_per_chunk": args.max_chars},
        "n_articles": len(out_rows), "output_sha256": sha256(out_csv), "device": device,
        "library_versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    (out_dir / "translation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.private_output_repo, repo_type="dataset", private=True, exist_ok=True, token=token)
    for p in (out_csv, out_dir / "translation_manifest.json"):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                        repo_id=args.private_output_repo, repo_type="dataset", token=token)
    print(f"Uploaded translation to {args.private_output_repo}", flush=True)


if __name__ == "__main__":
    main()
