# Changelog

## 1.2.0 (first public release, not yet published)

Contents: the annotation codebook, the two prompts, the adaptation and decoding settings, the
pipeline source from sampling to inference and the translation scripts, the analysis scripts, the de-identified result files,
`reproduce.py` with the values printed in the article and its supplement, six invented example
articles and the cards of the four LoRA adapters. The data include the two annotators' labels
before adjudication, the predictions on the machine-translated English articles and the two
sampled-decoding runs of Apertus-8B, so that inter-annotator agreement, the cross-lingual comparison
and the decoding check are recomputed as well.

Differences from what was run for the article:

- The three extractor runners check every output against the output schema, retry an invalid
  output with a larger token budget and count invalid outputs. A valid output is recorded as
  before, with two added fields, `validation_repairs` and `failed_attempts`.
- The LoRA trainer seeds all random number generators before the adapter is created.
- In `data/event_window_audit/event_window_audit_export.csv` the `annotator` field of the export,
  which named the account under which the labels were entered, is replaced by `label_source`,
  which reads `final_adjudicated`. The labels are unchanged.
- The descriptive notes in the four LoRA configuration files and the descriptions of two
  training-package scripts were reworded. Configuration keys, file paths and run identifiers are
  unchanged.

## 1.0.0 and 1.1.0

Internal versions, shared with the co-authors only and not published.
