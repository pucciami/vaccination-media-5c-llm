# Pipeline source

This folder holds the source that produced the models and predictions behind the article, copied
from the authors' working project. The scripts read the corpus and the annotation files from paths
under `data/`, which are not deposited. They show exactly what was done and can be run on another
corpus laid out the same way, but they do not run on this repository alone. What runs on the
repository alone is described in the top-level `README.md`.

Two changes were made to the source for this release. The three extractor runners
(`run_qwen_longitudinal_stage2_hf_job.py`, `run_qwen_merged_eval_hf_job.py` and
`run_apertus_adapter_eval_hf_job.py`) now check every output against the output schema, retry an
invalid output with a larger token budget and count invalid outputs. A valid output is recorded as
before, with two added fields, `validation_repairs` and `failed_attempts`. The LoRA trainer now seeds
all random number generators before the adapter is created.

Run the scripts from this `pipeline/` folder, so that imports such as
`from src.common.media_text import build_model_text` resolve. The two scripts in `src/analysis/`
import each other directly and are run from that folder. File names that begin with `apertus_` are
historical: `train_apertus_lora_media.py` and `prepare_apertus_sft_media.py` were used for both model
families, and the configuration file selects the family.

Dependencies are in `requirements-pipeline.txt`, which includes the GPU packages listed in
`requirements-training.txt`. Where the article used settings that differ from the script defaults,
they are listed in the top-level `README.md` under "Settings used in the article".

## Steps and files

| Step | What it does | File |
|---|---|---|
| 1 | Draws the 400-article sample: 300, then 100 more, stratified by six disease groups defined by keyword matching, with a cap per outlet | `src/labeling/make_labelstudio_articles_sample.py`, `src/labeling/make_labelstudio_articles_sample_plus100.py` |
| 2 | Splits the 400 adjudicated articles into 300 for training and 100 for the held-out evaluation set | `src/labeling/split_gold400_train_test.py` |
| 3 | Cleans obvious OCR and layout debris from the text; builds the model input from title and body only, without the press-clipping topics line | `src/eval/build_media_clean_v2_branch.py`, `src/common/media_text.py` |
| 4 | Builds the supervised training package from the 300 training articles | `src/labeling/prepare_clean_v2_gold300_train_package.py`, `src/models/prepare_apertus_sft_media.py` |
| 5 | Supervised LoRA adaptation of both families | `src/models/train_apertus_lora_media.py`, run through `src/models/run_apertus_hf_job.py`; settings in `configs/*_gold300_relevance_readj_v1.json` |
| 6 | Selects further articles for annotation (next section) | `src/labeling/prepare_media_active_learning_pool.py` to `src/labeling/prepare_active_learning_annotation_batches.py` |
| 7 | Moves the articles into and out of the annotation tool, merges the adjudicated labels and removes one duplicate (160 to 159) | `src/labeling/prepare_active_learning_labelstudio_import.py`, `parse_active_learning_labelstudio_export.py`, `prepare_active_learning_merge_manifest.py`, `apply_active_learning_holdout_resolution.py` |
| 8 | Builds the continuation package: the 159 new articles and 90 replayed training articles | `src/labeling/prepare_stage_b_continuation_package.py` |
| 9 | Continues the LoRA adaptation of both families from the step 5 adapter | as step 5; settings in `configs/*_stage_b_al159_replay90_v1.json` |
| 10 | Selects the 4,577 event-window candidates by the lexical rules in `../configs/lexical_candidate_rules.md` | `src/analysis/prepare_2021_inference_population.py`, `src/analysis/longitudinal_2021_common.py` |
| 11 | Runs the relevance classifier (base model) on the evaluation set | Qwen3-32B: `src/models/run_qwen_relevance_pool_sglang_hf_job.py`; Apertus-8B: `src/models/run_apertus_adapter_relevance_gate_hf_job.py` without an adapter |
| 12 | Runs the extractor on the evaluation set | Qwen3-32B: `src/models/run_qwen_merged_eval_hf_job.py`; Apertus-8B: `src/models/run_apertus_adapter_eval_hf_job.py` |
| 13 | Runs the extractor on the event-window candidates (the relevance classifier was not run on them) | `src/models/run_qwen_longitudinal_stage2_hf_job.py` |
| 14 | Machine-translates the 100 evaluation-set articles into English with NLLB-200-distilled-1.3B: sentence-boundary chunks of at most 480 characters, beam search with 5 beams, at most 512 new tokens per chunk; its input is the restricted Slovenian text, not deposited; the translations are then classified as in steps 11 and 12 | `src/translation/run_nllb_translation_hf_job.py`, run through `src/translation/launch_nllb_translation_hf_job.py` |
| 15 | Scores the evaluation set, standardises the event-window series and scores the audit | `../scripts/`, which run on the deposited files |

The other files in `src/models/` start, pause or resume hosted Hugging Face endpoints and merge
adapters into base weights. They are included so that the source is complete. The results in the
article come from the runs in the table.

The training loss was computed on the label output only. In `train_apertus_lora_media.py` every
prompt token, including the article text, is masked (`labels = [-100] * len(prompt) + assistant_ids`).
The training target is the label JSON object, which contains no article text.

## How the 159 further articles were selected

The selection was made once, with Qwen3-32B. The same 159 annotated articles were then used to
continue the adaptation of both model families. No separate selection was made with Apertus-8B.

Articles were selected by rules applied to the model's output. No token probability or other
uncertainty score was used. The steps, with the values set in the scripts:

1. **Pool.** All corpus articles except the 400 already annotated, matched on `source_doc_id`
   (`prepare_media_active_learning_pool.py`).
2. **Screening sample.** A sample of the pool, stratified by year and outlet: up to 3,500 articles
   with a vaccination term in the title, 700 with one in the lead only, and 300 with neither as
   controls (`prepare_relevance_candidate_tranche.py`, seed 42).
3. **Relevance classifier.** The base Qwen3-32B relevance classifier was run on the screening
   sample.
4. **Subset for the extractor.** Up to 600 articles, filled in this order: responses of the
   relevance classifier that could not be parsed (up to 20), articles judged relevant without a
   vaccination term in the lead (up to 150), articles with a phrase about deciding or weighing risks
   and benefits, which is the cue for Calculation (up to 180), the longest inputs (up to 120) and
   the slowest responses (up to 80). The rest was filled evenly across years
   (`prepare_targeted_stage2_exploratory_subset.py`, seed 42).
5. **Extractor.** A supervised Qwen3-32B extractor trained on the 300 training articles was run on the
   subset (`run_qwen_active_learning_stage2_hf_job.py`).
6. **Two annotation batches, 160 articles** (`prepare_active_learning_annotation_batches.py`,
   seed 42). Within each group below, articles were ranked by response time and then by input
   length, longest first. With reasoning on, the response time grows with the length of the model's
   reasoning, so the ranking favours articles the model spent longest on.
   - Calculation batch, 70 articles with a Calculation cue: 20 predicted supportive, 20 predicted
     neutral, 10 predicted undermining, and 20 in which the model found no Calculation. Any
     shortfall was filled at random.
   - Broad batch, 90 articles: unparsed responses (10), relevant without a vaccination term in the
     lead (25), predicted relevant but without 5C content (20), two or more constructs predicted
     (20), slowest responses (10) and longest inputs (10). Any shortfall was filled evenly across
     years.
7. **Annotation.** Both annotators annotated the 160 articles without seeing the model's labels,
   and disagreements were adjudicated under the same codebook. One article duplicated another and
   was removed, leaving 159.

The groups in steps 4 and 6 were chosen to catch cases the supervised model handled poorly and cases
where the relevance classifier and a simple keyword check disagreed. The Calculation batch was added
to give the model more examples of Calculation. The held-out evaluation set took no part in any step.

The numbers realised, from the manifests these scripts wrote: a pool of 35,752 articles, a screening
sample of 4,500 (3,500, 700 and 300), 3,044 parsed and 6 unparsed responses of the relevance
classifier when the subset was drawn, an extractor subset of 600 and annotation batches of 70 and
90. Calculation was present in 77 of the 159 articles (48%), against 29 of the 300 training
articles (10%). Collective Responsibility was present in 90 (57%) against 30 (10%), and two or more
constructs in 137 (86%) against 112 (37%).
