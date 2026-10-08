# Broad Vaccination Relevance Gate Prompt

Prompt version: `relevance-gate-v2-broad-vaccination-2026-05-14`

You are classifying one full media article for a PhD study on vaccination-related media discourse.

Return exactly one JSON object with these fields:
- `relevance`
- `reason_code`

Allowed values:
- `relevance`: `relevant`, `not_relevant`
- `reason_code`:
  - `explicit_vaccination_discussion`
  - `vaccination_policy_logistics_or_programme`
  - `vaccine_preventable_disease_with_vaccination_angle`
  - `disease_only_no_vaccination`
  - `public_health_general_no_vaccination`
  - `pandemic_or_institution_context_only`
  - `treatment_diagnostics_outbreak_politics_without_vaccine_angle`
  - `vaccination_only_passing_or_background_reference`
  - `unclear_default_relevant`

Core task:

Q1. Does the article discuss vaccines, vaccination, immunisation, vaccine policy, vaccine programmes, vaccine access, vaccine uptake, vaccine safety, vaccine effectiveness, vaccine hesitancy, or a vaccine-preventable disease in relation to vaccination?

Decision rule:
- `YES` -> `relevant`
- `NO` -> `not_relevant`

Broad relevance rule:
- Use `relevant` when vaccines or vaccination are discussed as part of the article's actual content from any point of view, even if the discussion is brief, factual, neutral, logistical, administrative, procurement-related, policy-related, critical, supportive, sceptical, or not opinionated.
- Do not require a strong 5C construct signal for relevance.
- A relevant article may later have `has_relevant_5c = no`.
- If the article discusses vaccination or vaccines at any length beyond a bare undeveloped word mention, let it through the gate.

Important positives:
- vaccination or vaccines directly
- vaccine uptake, refusal, hesitancy, acceptance, confidence, safety, effectiveness, benefit, risk, side effects, or misinformation
- vaccine access, availability, logistics, eligibility, cost, appointments, schedule, delivery, stock, procurement, donation, waste, recommendations, mandates, campaigns, or certificates
- disease risk, outbreak risk, or disease prevention explicitly connected to vaccination
- routine childhood vaccination, adult vaccination, booster vaccination, or vaccination programmes even when no single disease dominates

Important negatives:
- Disease names alone are insufficient.
- General health, prevention, infection, treatment, screening, hospitals, schools, sports, politics, or pandemic context alone are insufficient.
- Public-health institutions alone are insufficient.
- Use `not_relevant` only when vaccination is absent, purely incidental, or only a stock/background reference that is not developed in the article.

Boundary rule:
- If vaccination appears only as a bare undeveloped word mention, a list item, or an isolated background aside and the article's main content is clearly about something else, use `not_relevant`.
- If there is uncertainty between "bare mention" and "vaccination discussed as content", choose `relevant` and use `unclear_default_relevant`.

Choose `reason_code` as follows:
- `explicit_vaccination_discussion`: article discusses vaccination or vaccines directly
- `vaccination_policy_logistics_or_programme`: article discusses vaccine access, eligibility, delivery, stock, procurement, recommendations, certificates, mandates, campaigns, or programmes
- `vaccine_preventable_disease_with_vaccination_angle`: disease coverage explicitly connects disease risk or prevention to vaccination
- `disease_only_no_vaccination`: disease coverage without meaningful vaccination discussion
- `public_health_general_no_vaccination`: general health or prevention discussion without vaccination
- `pandemic_or_institution_context_only`: pandemic, institutional, NIJZ, government, or policy context without vaccination substance
- `treatment_diagnostics_outbreak_politics_without_vaccine_angle`: treatment, diagnostics, outbreak, or politics without vaccine angle
- `vaccination_only_passing_or_background_reference`: vaccine mention is incidental, stock, or undeveloped background
- `unclear_default_relevant`: borderline case where explicit vaccination content exists but may be brief; defaulted to relevant

Return JSON only. Do not explain. Do not add extra keys.

Article:

{{ARTICLE_TEXT}}
