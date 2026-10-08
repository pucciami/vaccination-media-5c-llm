# 5C classification of vaccination-related media articles with open-weight LLMs

[![reproduce](https://github.com/pucciami/vaccination-media-5c-llm/actions/workflows/reproduce.yml/badge.svg)](https://github.com/pucciami/vaccination-media-5c-llm/actions/workflows/reproduce.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22912892.svg)](https://doi.org/10.5281/zenodo.22912892)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Docs and data: CC BY 4.0](https://img.shields.io/badge/docs%20and%20data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DOCS)

Code, prompts, codebook and de-identified result files for the article *From annotation towards
surveillance: adapted open-weight large language models for article-level 5C classification of
vaccination-related media discourse, Slovenia, 2010–2024* by Mario Fafangel, Janez Žibert and
Lijana Zaletel-Kragelj (manuscript for Eurosurveillance).

The article describes a two-stage pipeline. A relevance classifier decides whether a news article
is about vaccination. An extractor then records which of the 5C constructs the article contains
(Confidence, Complacency, Constraints, Calculation, Collective Responsibility) and the valence of
each. Two open-weight model families, Qwen3-32B and Apertus-8B, were adapted with LoRA and one
round of active learning, then scored once on a held-out evaluation set of 100 double-annotated articles.
The better-performing family was chosen on that same set; there is no external confirmatory set.

```
pip install -r requirements.txt
python reproduce.py
```

`reproduce.py` recomputes the results from `data/` and checks them against the values printed in
the article and its supplement. GitHub Actions runs it on every push.

Status: pre-release (v1.2.0). The Zenodo DOI, 10.5281/zenodo.22912892, is reserved and resolves once the record is
published.

## Contents

```
reproduce.py      recomputes the published numbers and checks them
expected/         the values printed in the article and its supplement
requirements.txt  what reproduce.py needs (pandas, numpy)
data/             de-identified result files and labels, no article text
scripts/          the analysis scripts that reproduce.py runs
codebook/         the annotation codebook (Supplementary Methods S1)
prompts/          the two classification prompts, verbatim
configs/          adaptation and decoding settings; lexical rules of the event-window analysis
pipeline/         the pipeline source as run, from sampling to inference (see pipeline/README.md)
examples/         six invented articles that show the input format
model_cards/      cards of the four LoRA adapters released on Hugging Face
LICENSE           MIT, for all code
LICENSE-DOCS      CC BY 4.0, for documentation, codebook, prompts and data
CITATION.cff      citation metadata (.zenodo.json holds the Zenodo metadata)
CHANGELOG.md
```

## Reproducing the published numbers

`reproduce.py` runs the scripts below and compares their output with
`expected/headline_numbers.json`. They can also be run on their own:

```
# Evaluation-set estimates and confidence intervals (main-text tables; Supplementary Tables S1, S3, S4)
python scripts/compute_locked_performance_cis.py --output-dir out/

# Event-window z-scores, prediction limits and one-sided p-values (Supplementary Table S6)
python scripts/recompute_event_window_zscores.py --output-dir out/event_window

# Event-window audit precision
python scripts/score_event_window_audit.py data/event_window_audit/event_window_audit_export.csv

# Inter-annotator agreement (Supplementary Table S2)
python scripts/compute_annotator_agreement.py --output-dir out/annotator_agreement

# Slovenian against machine-translated English (Supplementary Table S5)
python scripts/compute_locked_performance_cis.py --matrix-dir data/cross_lingual \
    --paired-comparison "EN=data/cross_lingual/G0_E2_EN/G0_E2_EN_comparison.csv::SL=data/locked_set_comparisons/G0_E2/G0_E2_comparison.csv" \
    --output-dir out/cross_lingual

# Apertus-8B with sampled decoding (Supplementary Methods S3); reproduce.py also passes one
# --paired-comparison per seed against the greedy system A0_F0
python scripts/compute_locked_performance_cis.py --matrix-dir data/decoding_check --output-dir out/decoding_check
```

A run takes a minute or two, mostly for the bootstraps (10,000 resamples). Each bootstrap takes its
seed from a SHA-256 hash of the base seed and a fixed label, so an interval comes out the same
whether all six systems are run or only some.

What `reproduce.py` recomputes:

- every evaluation-set estimate and its confidence interval for all six systems: relevance accuracy,
  sensitivity, specificity and precision, any-5C and disease accuracy, exact match, per-construct
  F1, construct macro-F1 and micro-F1, and the mean F1 of the four constructs other than Calculation
  (computed, no longer reported in the article)
- the paired bootstrap differences between systems: the gain from active learning, the gap between
  Qwen3-32B and Apertus-8B, and each adaptation step within a family
- valence accuracy against the majority-class baseline (Supplementary Table S3), and recall and
  precision for undermining valence, per construct and pooled (Supplementary Table S4)
- the event-window z-scores and threshold bands for 13 indicators over 22 weeks, the number of
  alarms outside the event window, the one-sided prediction limits (95%, 99%, 99.9%, and a
  Bonferroni limit for 26 indicator-weeks) and the one-sided p-values (Supplementary Methods S5,
  Supplementary Table S6)
- the event-window audit: the precision of both flags, the unflagged comparison and the per-week counts
- the relevance routing counts (73 of 100 articles passed in each language; the Apertus-8B classifier's 25
  false passes, 11 of which received construct labels)
- inter-annotator agreement before adjudication, with its intervals (Supplementary Table S2)
- the comparison of Slovenian with machine-translated English (Supplementary Table S5)
- macro-F1 of the base Apertus-8B extractor with sampled decoding, for both seeds (Supplementary
  Methods S3)

Every value in `expected/headline_numbers.json` is typed from the article and its supplement as printed,
not from the recomputed output. Some printed numbers cannot be recomputed from this repository, because
they need the restricted corpus or labels that are not deposited: the corpus size, outlets and vaccine-term
share (36,152 articles, 501 outlets); the sampling flow and the active-learning selection and enrichment
counts (Supplementary Methods S3); the event-window candidate counts (4,577, 4,469 and 4,458); and the
agreement between the two audit coders (κ 0.81, 0.69 and 0.40), whose separate codings are available from
the authors on request.

## Data files

**`data/locked_set_comparisons/<system>/<system>_comparison.csv`** has one row per evaluation-set article
for each system. Every field has three columns: the adjudicated reference label, the model's
prediction and whether they match. `doc_id` is the article's number within the 400-article
annotated sample. The link from `doc_id` to the article is part of the restricted corpus.

A system code gives the model family (`G` Qwen3-32B, `A` Apertus-8B), the relevance classifier
(`0`, the base model in every system, since it was not adapted by design) and the adaptation level
of the extractor (`E0`/`F0` base, `E1`/`F1` supervised LoRA, `E2`/`F2` active learning).

How the scripts score:

- Construct F1 is scored end to end. If the relevance classifier drops an article, each construct
  in its reference label counts as a false negative.
- Any-5C and disease accuracy are computed over the 74 reference-relevant articles. A dropped
  article has no predicted disease and counts as wrong.
- Valence accuracy is computed over the articles whose reference label has the construct, with
  three classes. An article the pipeline did not code for that construct counts as wrong.
- Undermining recall is correct / reference-undermining and precision is correct /
  predicted-undermining (valence code −1). Per construct, the intervals are Wilson intervals. The
  pooled values add the counts over the five constructs. Because one article can carry several
  undermining labels, the pooled interval is an article-level percentile bootstrap.
- The one-sided p-value of an event week is the upper tail of t(6) at z / √(1 + 1/7). The
  Bonferroni-adjusted value is that p-value multiplied by 26.

**`data/event_window/longitudinal_2021_zscores_all_indicators_v3.csv`** is in long format, one row
per indicator per week. It covers 22 weeks starting on Mondays, from the week commencing 2 August to
the week commencing 27 December 2021. The last week has five days. The corpus starts on Sunday
1 August, which falls in a partial week and is left out.

| Column | Content |
|---|---|
| `period` | week commencing |
| `indicator_id`, `family` | the indicator and its family (presence, undermining or control) |
| `value` | weekly prevalence, or the article count for `article_volume` |
| `is_baseline` | one of the seven baseline weeks |
| `is_signal_window` | one of the two event weeks, commencing 27 September and 4 October |
| `is_context_week` | the week commencing 20 September, left out of both |
| `phi_raw`, `phi_used` | overdispersion diagnostic |
| `z_ascoded`, `z_baseUnc`, `z_quasi` | other standardisations, not used in the article |
| `z_ears`, `ears_band` | the statistic reported in the article and its threshold band |

The companion file `qwen_e2_longitudinal_2021_aug_dec_stage2_full_v1_weekly_indicators.csv` gives the
weekly counts behind the presence and control indicators. For the undermining indicators only the
weekly prevalence (`value`) is given, because their article-level inputs are restricted. `z_ears`
can therefore be recomputed for all 13 indicators, but the undermining prevalences cannot be rebuilt
from counts.

There are two control series. The share of articles that name the Janssen vaccine
(`janssen_lexical_share`) is a positive control that does not depend on the 5C classification. It
is computed among the articles the model assigned to COVID-19. Article volume (`article_volume`)
shows whether a shift could be a change in how much was published.

**`data/event_window_audit/event_window_audit_export.csv`** holds the final adjudicated labels for
all 167 articles that the model flagged in the event window, without article text or outlet. Two
annotators coded every article independently and without seeing the model's prediction, and their
disagreements were resolved as for the reference standard (consensus, then a third specialist). The columns are the
adjudicated labels (`expert_*`, where the valence value `neutral` is the codebook's
non-directional class), the model's flags (`model_*`) and `label_source`, which reads
`final_adjudicated` in every row. The two separate codings, from which agreement was computed, are
held by the authors.

**`data/annotator_agreement/annotator_labels_400.csv`** has the two annotators' labels for the 400
annotated articles before adjudication: relevance, primary disease, any-5C presence, the constructs
present and the valence of each (`ann1_*` and `ann2_*`). It holds label fields only, without article
text, notes or free-text fields. `doc_id` is the same article number as in the comparison files.

**`data/cross_lingual/G0_E2_EN/G0_E2_EN_comparison.csv`** has the same layout as the evaluation-set
comparison files. It holds the predictions of the Qwen3-32B pipeline on the 100 evaluation-set articles
after machine translation into English with NLLB-200, scored against the same reference labels. The
translations themselves are not deposited; the translation scripts are in `pipeline/src/translation/`
(NLLB-200 is licensed CC BY-NC 4.0).

**`data/decoding_check/<system>/<system>_comparison.csv`** holds the base Apertus-8B extractor run
with the sampling settings recommended on its model card (temperature 0.8, top_p 0.9) instead of
greedy decoding, for two seeds (`S20260417` and `S7`). The predictions went through the base Apertus-8B
relevance classifier exactly as the greedy system `A0_F0`, which uses the same route and is rebuilt
identically by it. In seed 20260417 one output (article 295) could not be parsed. It is scored as a
prediction with no construct.

**`configs/lexical_candidate_rules.md`** gives the regular expressions and the span of text that
defined the 4,577 event-window candidates and the Janssen control.

## The pipeline

`pipeline/` holds the source that produced the models and predictions, from drawing the sample to
running the extractor. `pipeline/README.md` lists the steps in order with the file for each, and
describes how the 159 active-learning articles were selected. The scripts read the corpus and the
annotation files, which are not deposited, so they document what was done and can be run on another
corpus, but not on this repository alone.

### Where it ran

All training and inference for the article ran as private Hugging Face Jobs under the authors'
Hugging Face account, on Hugging Face GPUs (A100 and H200 for Qwen3-32B). For these jobs the article
texts were held in private dataset repositories of the same account. No proprietary model service was
used. Each `launch_*.py` file in `pipeline/src/models/` submits the matching `run_*.py` file as a job.
The `run_*.py` files also run directly on any machine with a suitable GPU, which is how the commands
below use them.

### Running the pipeline on your own articles

`examples/synthetic_articles.csv` holds six invented Slovenian articles without reference labels.
They show the input format: `doc_id`, `title`, `text`, and `model_text` as built by
`pipeline/src/common/media_text.py`. Put your own articles in the same columns. You need a GPU that
can hold the model.

```
pip install -r pipeline/requirements-pipeline.txt
```

`pipeline/requirements-pipeline.txt` lists every package the pipeline imports, including
`scikit-learn`, `openpyxl`, `pyarrow`, `httpx` and, on Linux, `sglang` for the relevance
classifier.

```
# Extractor with the adapter used in the article (active learning)
python pipeline/src/models/run_qwen_longitudinal_stage2_hf_job.py \
    --base-model Qwen/Qwen3-32B \
    --adapter-repo Pucciami/vaccination-media-5c-qwen3-32b-lora \
    --input-csv examples/synthetic_articles.csv \
    --system-prompt-path prompts/stage2_extractor_system_prompt_v7.txt \
    --output-dir out/stage2 \
    --private-output-repo <YOUR_HF_USER>/<YOUR_OUTPUT_DATASET>

# Extractor, base model without an adapter (--base-model works only with --adapter-repo)
python pipeline/src/models/run_qwen_longitudinal_stage2_hf_job.py \
    --model-repo Qwen/Qwen3-32B \
    --input-csv examples/synthetic_articles.csv \
    --system-prompt-path prompts/stage2_extractor_system_prompt_v7.txt \
    --output-dir out/stage2_base \
    --private-output-repo <YOUR_HF_USER>/<YOUR_OUTPUT_DATASET>

# Relevance classifier with the settings used in the article (starts a local sglang server)
python pipeline/src/models/run_qwen_relevance_pool_sglang_hf_job.py \
    --model-repo Qwen/Qwen3-32B-FP8 \
    --input-csv examples/synthetic_articles.csv \
    --prompt-path prompts/relevance_gate_prompt_v2.md \
    --schema-path pipeline/configs/apertus_relevance_gate_schema_v2_broad_vaccination.json \
    --enable-thinking --temperature 0.0 --top-p 1.0 --presence-penalty 0.0 --max-new-tokens 512 \
    --output-dir out/stage1 \
    --private-output-repo <YOUR_HF_USER>/<YOUR_OUTPUT_DATASET>
```

The scripts write predictions to `--output-dir` and upload checkpoints to the Hugging Face dataset
named in `--private-output-repo`. They therefore need a Hugging Face token with write access in the
`HF_TOKEN` environment variable. The extractor scripts accept an output only if it follows the
output schema of the extractor prompt: every field present, `disease_1` from the allowed list, and a
valence for each construct present and none for a construct absent. An invalid output is tried again
with a larger token budget, and the run manifest counts invalid, retried and failed outputs. To
adapt a model to your own annotated articles, follow steps 4, 5, 8 and 9 in `pipeline/README.md`.

### Settings used in the article

The script defaults were left as they were during the runs. Where a default differs from what the
article used, pass the article's setting explicitly.

| Step | Script | Setting used in the article | Script default, where different |
|---|---|---|---|
| Relevance classifier (Qwen3-32B) | `run_qwen_relevance_pool_sglang_hf_job.py` | `Qwen/Qwen3-32B-FP8` served with sglang and the qwen3 reasoning parser, thinking mode on (`--enable-thinking`), temperature 0.0, top_p 1.0, presence penalty 0.0, `max_new_tokens` 512 | temperature 0.6, top_p 0.95, presence penalty 0.5, thinking mode off |
| Extractor, evaluation set (Qwen3-32B) | `run_qwen_merged_eval_hf_job.py` with `--base-model`, `--adapter-repo` and `--force-relevant-output` | greedy decoding (`do_sample=False`), `max_new_tokens` 1024, retried at 2048 when the output was truncated or invalid (`--max-new-tokens 1024 --retry-max-new-tokens 2048`), thinking mode on (the Qwen3 default), reasoning text discarded | `max_new_tokens` 256, no retry |
| Extractor, event window and new articles (Qwen3-32B) | `run_qwen_longitudinal_stage2_hf_job.py` | greedy decoding, `max_new_tokens` 1024, retried at 2048 and then 4096 | none |
| Extractor (Apertus-8B) | `run_apertus_adapter_eval_hf_job.py` | greedy decoding (`do_sample=False`, temperature 0.0, top_p 1.0), no reasoning mode | none |

Supplementary Methods S3 ("Serving and decoding configuration") gives the same settings.
`configs/lora_adaptation_config.md` gives the adaptation settings of both stages and both families.

## Released adapters

The four LoRA adapters reported in the article, one for each adapted system in Supplementary Table
S1, are on Hugging Face. Their model cards are copied in `model_cards/`. Both base models are openly
available under Apache-2.0: `Qwen/Qwen3-32B` and `swiss-ai/Apertus-8B-Instruct-2509`.

| System | Repository |
|---|---|
| Qwen3-32B, active learning (the extractor used in the article) | `Pucciami/vaccination-media-5c-qwen3-32b-lora` |
| Qwen3-32B, supervised LoRA | `Pucciami/vaccination-media-5c-qwen3-32b-lora-supervised` |
| Apertus-8B, active learning (the comparator) | `Pucciami/vaccination-media-5c-apertus-8b-lora` |
| Apertus-8B, supervised LoRA | `Pucciami/vaccination-media-5c-apertus-8b-lora-supervised` |

Retraining with the code here is seed-controlled, but GPU kernels are not fully deterministic, so a
retrained adapter may differ slightly from the released one. The released adapters are the ones to
use for an exact copy of the article's models.

## Restricted data

The article texts cannot be shared. They belong to an institutional press-clipping corpus held by
the National Institute of Public Health Slovenia (NIJZ) and are subject to the terms of the
press-clipping service. This repository contains no article text. Upstream prediction
outputs and the two separate audit codings are available from the corresponding author on reasonable request.

## Names used in the files

Run and file names were kept as they were during the work, so that they match the authors'
records. The main ones:

| Name | Meaning |
|---|---|
| `relevance-gate-v2-broad-vaccination-2026-05-14` | the relevance-classifier prompt, version 2, which leans towards keeping articles, fixed on 14 May 2026 (`prompts/relevance_gate_prompt_v2.md`) |
| extractor system prompt v7 | the 5C extractor prompt, version 7 (`prompts/stage2_extractor_system_prompt_v7.txt`) |
| `qwen3-32b-5c-media-lora-clean-v2-stage-b-al159-replay90-v1` | the Qwen3-32B active-learning adapter: `stage-b` is the active-learning stage, `al159` the 159 active-learning articles and `replay90` the 90 replayed training articles |
| `qwen_e2_longitudinal_2021_aug_dec_stage2_full_v1` | the extractor run over all event-window candidates, August to December 2021 (Supplementary Methods S5) |
| `gold100`, `gold300`, `gold400` | in code: the held-out evaluation set ("locked" in file and script names), the training set and the whole annotated sample |
| gate | in file and argument names: the relevance classifier |
| `apertus_` | at the start of a prompt, schema or training file name: historical; the file was used for both model families |

## Licences

- Code (`scripts/`, `pipeline/` apart from `pipeline/README.md`, `reproduce.py`,
  `examples/build_example_input.py`, `requirements.txt` and `.github/`): MIT, see `LICENSE`.
- Documentation, codebook, prompts and data (this README and the other documentation, `codebook/`,
  `prompts/`, `configs/`, `model_cards/`, `expected/`, `data/` and
  `examples/synthetic_articles.csv`): CC BY 4.0, see `LICENSE-DOCS`.
- The model weights are not in this repository. The adapters on Hugging Face carry the licence of
  their base model, Apache-2.0 for both.

## How to cite

Cite the Zenodo record and the article. `CITATION.cff` has the metadata.

> Fafangel M, Žibert J, Zaletel-Kragelj L. Code and data for "From annotation towards surveillance:
> adapted open-weight large language models for article-level 5C classification of
> vaccination-related media discourse, Slovenia, 2010–2024". Version 1.2.0. Zenodo; 2026.
> doi:10.5281/zenodo.22912892.
