---
license: apache-2.0
base_model: Qwen/Qwen3-32B
library_name: peft
pipeline_tag: text-generation
language:
- sl
tags:
- lora
- peft
- vaccination
- public-health
- media-monitoring
- 5c-framework
---

# Qwen3-32B LoRA adapter for 5C classification of vaccination-related media articles (active learning)

A LoRA adapter for `Qwen/Qwen3-32B`. It reads one full-text news article about vaccination and returns a
JSON object with the primary disease, the 5C constructs present in the article (Confidence,
Complacency, Constraints, Calculation, Collective Responsibility) and the valence of each:
supportive, non-directional or undermining of vaccination. Adaptation stage: active learning. This is the extractor used in the article's pipeline.

The adapter is the second stage of a two-stage pipeline. First, a relevance classifier (the base
model with a prompt, no adapter) decides whether an article is about vaccination. Only relevant
articles are passed to this extractor.

- Article: Fafangel M, Žibert J, Zaletel-Kragelj L. From annotation towards surveillance: adapted open-weight large language models for article-level 5C classification of vaccination-related media discourse, Slovenia, 2010–2024. Manuscript for Eurosurveillance.
- Code, prompts, codebook and result files: Zenodo, doi:10.5281/zenodo.22912892
- Run identifier in the article's supplement: `qwen3-32b-5c-media-lora-clean-v2-stage-b-al159-replay90-v1`
- Language: the training articles are Slovenian. In the article's sensitivity analysis, which covered
  the deployed Qwen3-32B pipeline only, performance on machine-translated English was similar.

## What the labels mean

The labels describe how an article frames vaccination. They are media framing indicators for
monitoring by experts. They do not measure the attitudes or behaviour of any person or population,
and they do not support causal statements about vaccination uptake.

## Performance

Scored once on a held-out evaluation set of 100 articles, double-annotated and adjudicated, that was
never used for adaptation. The same set was used to choose between the two model families; there is no
external confirmatory set. The scores are end to end, with the base relevance classifier of the same
model family (system `G0_E2`). The 95% confidence intervals are Wilson score intervals for
accuracy and article-level percentile bootstrap intervals (10,000 resamples) for F1.

| Metric | Value (95% CI) |
|---|---|
| Relevance accuracy (relevance classifier) | 0.970 (0.915–0.990) |
| Construct-presence macro-F1 | 0.752 (0.657–0.815) |
| Construct-presence micro-F1 | 0.809 (0.759–0.852) |

Per-construct F1 of the deployed Qwen3-32B pipeline is in Table 2 of the article, and its valence
results in Supplementary Tables S3 and S4. `python reproduce.py` in the Zenodo record recomputes the
per-construct and valence results of every system, this one included.

Limits. The evaluation rests on one held-out evaluation set of 100 articles from one national media system, with
no external set for confirmation. Confidence intervals are wide for the rarer constructs, and
Calculation is exploratory. Undermining valence is detected poorly. For the deployed Qwen3-32B
pipeline, pooled recall and precision were both 0.32 (6/19), with 95% confidence intervals of
0.08–0.56 and 0.09–0.52 (article-level bootstrap). Check the output before acting on it.

## Training

- Data: continued from the supervised adapter on 159 further annotated articles and 90 replayed training articles, 249 in all, of which 224 were trained on and 25 held out (2 epochs, learning rate 1e-4). All articles come from a Slovenian mainstream-media press-clipping corpus
  (2010–2024) and were annotated by two public-health specialists under the codebook in the Zenodo
  record. The articles cannot be shared.
- LoRA r 16, alpha 16, dropout 0.05 on the attention and MLP projections (q, k, v, o, gate, up,
  down), 4-bit base weights, seed 20260417. The full configuration is in `training_config_used.json`
  in this repository and in `pipeline/configs/qwen3_32b_lora_media_clean_v2_stage_b_al159_replay90_v1.json` in the Zenodo record.
- The loss was computed on the label JSON only. All prompt tokens, including the article text, were
  masked, and the training target contains no article text.
- The settings were fixed in advance, with no hyperparameter search. The held-out articles were used
  only to watch the loss after each epoch. No checkpoint was selected on them.
- Trained as a private Hugging Face Job under the authors' account. Retraining with the code in the
  Zenodo record is seed-controlled, but GPU kernels are not fully deterministic, so a retrained
  adapter may differ slightly. Use this adapter for an exact copy.

## Use

Use the extractor system prompt from the Zenodo record unchanged
(`prompts/stage2_extractor_system_prompt_v7.txt`). The user message is `Article:`, a line break, the
article title, a blank line and the article body. Decode greedily. The article's runs used `max_new_tokens` 1024 and retried at 2048, and at 4096 in the event-window run, when the output was truncated or not valid JSON. Qwen3-32B ran with its reasoning mode on. Text inside `<think>` tags is discarded and only the final JSON object is used.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen3-32B"
adapter = "Pucciami/vaccination-media-5c-qwen3-32b-lora"
tokenizer = AutoTokenizer.from_pretrained(base)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto"), adapter
)

system_prompt = open("stage2_extractor_system_prompt_v7.txt", encoding="utf-8").read()
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Article:\n" + title + "\n\n" + body},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)
output = model.generate(inputs, do_sample=False, max_new_tokens=1024)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

The script used for the article's runs, with JSON parsing, retries and checkpoints, is
`pipeline/src/models/run_qwen_longitudinal_stage2_hf_job.py` in the Zenodo record.

## Citation

Fafangel M, Žibert J, Zaletel-Kragelj L. From annotation towards surveillance: adapted open-weight large language models for article-level 5C classification of vaccination-related media discourse, Slovenia, 2010–2024. Manuscript for Eurosurveillance. Code and data: doi:10.5281/zenodo.22912892.

## Licence

Apache-2.0, the licence of the base model `Qwen/Qwen3-32B`. This is one of the four adapters released with
the Zenodo record of the article.
