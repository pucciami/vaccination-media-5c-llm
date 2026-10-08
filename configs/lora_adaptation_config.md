# Adaptation and decoding settings

## Base models

Both base models are openly available under Apache-2.0.

- Qwen3-32B: `Qwen/Qwen3-32B`. The relevance classifier was served from the FP8 release,
  `Qwen/Qwen3-32B-FP8`.
- Apertus-8B: `swiss-ai/Apertus-8B-Instruct-2509`, an open-weight model developed in Switzerland.

All training and inference ran as private Hugging Face Jobs under the authors' account (see
"Where it ran" in the top-level `README.md`).

## Decoding

Decoding was deterministic throughout, so no generation depended on a random seed.

- Relevance classifier, Qwen3-32B: served with sglang and the qwen3 reasoning parser, thinking mode
  on, temperature 0.0, top_p 1.0, `max_new_tokens` 512. Only the final JSON object was used and the
  reasoning text was discarded.
- Extractor, Qwen3-32B: Hugging Face Transformers, greedy decoding (`do_sample=False`), thinking mode
  on, `max_new_tokens` 1024. Again only the final JSON object was used. A truncated or invalid
  output was generated again with 2048 tokens, and in the event-window run also with 4096, because
  the reasoning text comes before the JSON object and can use up a small budget. The event-window run
  (`qwen_e2_longitudinal_2021_aug_dec_stage2_full_v1`, 27 May 2026) used the same adapter and v7
  prompt as the evaluation-set run and classified all 4,577 candidate articles without a generation
  error.
- Extractor, Apertus-8B: the same Transformers code, greedy decoding (`do_sample=False`, temperature
  0.0, top_p 1.0), no reasoning mode.

## LoRA settings, both stages and both families

- r 16, alpha 16, dropout 0.05, on the q, k, v, o, gate, up and down projections.
- 4-bit (NF4) base weights, batch size 1 with 8 gradient-accumulation steps, warmup ratio 0.05,
  weight decay 0.01, optimiser `paged_adamw_8bit`, seed 20260417.
- The settings were fixed in advance, with no hyperparameter search.
- Each training package holds out 10% of its articles, chosen by a seeded hash of the article
  identifier. This development subset was used only to watch the loss after each epoch. No
  checkpoint or model was selected on it.
- The trainer seeds Python, NumPy, PyTorch and Transformers before the adapter is created.
  Retraining is therefore seed-controlled, but GPU kernels are not fully deterministic, so a
  retrained adapter may differ slightly from the released one.

## The two stages

Supervised stage: 3 epochs, learning rate 2e-4, on the 300 training articles, of which 270 were
trained on and 30 held out.

Active-learning stage: 2 epochs, learning rate 1e-4, continued from the supervised adapter on 249
articles, of which 224 were trained on and 25 held out. The 249 are 159 further articles and 90
replayed training articles, which were included so that the model would not forget what it had
learnt from the training set. The 159 were chosen once, from the unlabelled corpus, by rules applied
to Qwen3-32B output, and no model probability was used (details in `pipeline/README.md`). 160
articles were selected, annotated by both annotators without seeing the model's labels, and
adjudicated under the same codebook. One duplicate was removed. The same 159 articles were used for
both families.

The extractor was evaluated at three adaptation levels in each family: base, supervised LoRA and
active learning. The relevance classifier was the base model throughout (Supplementary Table S1).

| Stage | System | Training articles | Epochs | Learning rate | Run identifier | Released as |
|---|---|---|---|---|---|---|
| Supervised LoRA | G0_E1 | 300 training articles (270 trained, 30 held out) | 3 | 2e-4 | `qwen3-32b-5c-media-lora-clean-v2-gold300-readj-v1` | `Pucciami/vaccination-media-5c-qwen3-32b-lora-supervised` |
| Active learning | G0_E2 | continued from G0_E1 on 159 new and 90 replayed articles (224 trained, 25 held out) | 2 | 1e-4 | `qwen3-32b-5c-media-lora-clean-v2-stage-b-al159-replay90-v1` | `Pucciami/vaccination-media-5c-qwen3-32b-lora` |
| Supervised LoRA | A0_F1 | 300 training articles (270 trained, 30 held out) | 3 | 2e-4 | `apertus-5c-media-lora-clean-v2-gold300-readj-v1` | `Pucciami/vaccination-media-5c-apertus-8b-lora-supervised` |
| Active learning | A0_F2 | continued from A0_F1 on the same 159 new and 90 replayed articles (224 trained, 25 held out) | 2 | 1e-4 | `apertus-5c-media-lora-clean-v2-stage-b-al159-replay90-v1` | `Pucciami/vaccination-media-5c-apertus-8b-lora` |

G0_E2 is the extractor used in the article's pipeline. The run identifiers are the names used in the
article's supplement and in the authors' records. The configuration files as run are
`pipeline/configs/*_gold300_relevance_readj_v1.json` (supervised stage) and
`pipeline/configs/*_stage_b_al159_replay90_v1.json` (active-learning stage). The training articles
contain text from the restricted corpus and are not shared. The per-article comparison files for all
six systems are in `data/locked_set_comparisons/`.

## Translation for the cross-lingual check

`facebook/nllb-200-distilled-1.3B`, deterministic beam search (`num_beams` 5, `do_sample=False`),
input in chunks of at most 1024 tokens. The translation was separate from the classifier.
