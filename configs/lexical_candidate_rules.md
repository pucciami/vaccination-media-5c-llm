# Lexical rules of the event-window analysis

The rules are applied to the article title, a newline and the first 2,100 characters of the model
text, which itself starts with the title and a blank line before the body, so slightly fewer than
2,100 body characters are searched (as run; `longitudinal_2021_common.lexical_flags`). Matching is case-insensitive (Python `re.search` with `re.IGNORECASE`).

| Rule | Regular expression | Role |
|---|---|---|
| COVID-19 term | `covid|koronavirus|sars[- ]?cov[- ]?2` | candidate filter, required |
| Vaccination term | `cepiv|cepljen|cepljiv|cepljenj|precep|vakcin|imuniz|vaccin` | candidate filter, required |
| Janssen vaccine | `janssen|johnson\s*&\s*johnson|j\s*&\s*j` | positive control (`janssen_lexical_share`) |

An article published from 1 August to 31 December 2021 became an event-window candidate if it
matched both the COVID-19 rule and the vaccination rule. There were 4,577 candidates. The relevance
classifier was not run on this set (see the article's Methods). The extractor assigned each
candidate a primary disease. Of the 4,469 assigned to COVID-19, the 11 published on Sunday 1 August
2021 fall in a partial week that is left out of the weekly series. The remaining 4,458 articles are
the denominator of the weekly prevalences.

The Janssen control does not depend on the 5C classification. It is computed among the articles the
model assigned to COVID-19.

The rules are defined in `pipeline/src/analysis/longitudinal_2021_common.py` (`COVID_REGEX`,
`VACCINATION_REGEX`, `JANSSEN_REGEX`, `DEFAULT_LEXICAL_SEARCH_CHARS = 2100`) and applied by
`pipeline/src/analysis/prepare_2021_inference_population.py`.
