# Changelog

## 0.6.0 — Phase 6

- Added paired standard-versus-masked fairness sensitivity auditing.
- Added careful masking for known names, contacts, pronouns, honorifics, and institution names.
- Preserved structured skills, experience, degrees, fields, and certifications during masking.
- Added diagnostic score-component and rank-change reporting without altering original decisions.
- Added small-cohort warnings and robust median/MAD threshold estimation for larger cohorts.
- Added human-review-only dispositions and explicit limitations on fairness claims.
- Added the `fairness-audit` CLI workflow and reviewed threshold overrides.
- Pinned scikit-learn 1.8.0 to match the included serialized demo model.
- Expanded the full regression suite from 44 to 50 passing tests.

## 0.5.0 — Phase 5

- Added a taxonomy-validated 55-node directed skill-prerequisite graph.
- Added cycle, missing-prerequisite, self-link, and positive-hours validation.
- Added prerequisite-aware learning paths and rough planning-hour estimates.
- Added mandatory/preferred/relevant recommendation priorities.
- Added individual and combined conservative score simulations.
- Added eligibility-after-learning checks using the existing Phase 2 engine.
- Added the `growth` CLI workflow and custom-graph support.
- Expanded the full regression suite from 36 to 44 passing tests.

## 0.4.0 — Phase 4

- Added calibrated Linear SVM probabilities using three-fold sigmoid calibration.
- Added validated and normalized configurable composite-score weights.
- Added exact per-component contributions and reproducible arithmetic explanations.
- Added model-inferred versus user-overridden target-role provenance.
- Added low-confidence job-role warnings and unknown-role rejection.
- Preserved Phase 2 mandatory eligibility as the first ranking rule.
- Expanded the full regression suite from 31 to 36 passing tests.

## 0.3.0 — Phase 3

- Added labelled CSV validation, provenance metadata, duplicate removal, and conflict checks.
- Added deterministic stratified train/validation/test splitting with leakage assertions.
- Added TF-IDF Logistic Regression, Linear SVM, and Random Forest pipelines.
- Added validation-based model selection and one-time held-out test evaluation.
- Added saved model bundles, confusion matrices, predictions, split manifests, and model cards.
- Added role prediction and hybrid scoring CLI commands.
- Added removal of known identity and contact fields before ML inference.
- Added a clearly labelled synthetic demonstration dataset.
- Expanded the full regression suite from 22 to 31 passing tests.

## 0.2.0 — Phase 2

- Added structured job-description parsing
- Added mandatory, preferred, and relevant skill classification
- Added minimum-experience, degree, field-of-study, and certification requirements
- Added dynamic weighted requirement scoring
- Added mandatory eligibility independent of ranking score
- Added candidate ranking and detailed explanations
- Added `parse-job` and `match` command-line workflows
- Added LinkedIn, GitHub, portfolio, PDF-link, and DOCX-link extraction
- Added a second sample candidate and sample job description
- Expanded the automated test suite to 22 tests

## 0.1.0 — Phase 1

- Added PDF, DOCX, and TXT extraction
- Added resume-safe cleaning and section detection
- Added identity, skills, education, and experience extraction
- Added canonical skills taxonomy and privacy-aware JSON output
- Added the original `parse` command and automated tests
