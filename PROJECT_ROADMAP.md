# CareerFit Implementation Roadmap

## Current verified status

- Phase 1: completed and regression-tested.
- Phase 2: completed and regression-tested.
- Phase 3: completed as an ML pipeline and verified with synthetic demonstration data.
- Phase 4: completed as an auditable composite-scoring engine.
- Phase 5: completed as a conservative graph-backed growth-planning engine.
- Phase 6: completed as a paired masking and score-sensitivity audit.
- Real-world Phase 3 performance remains unverified until a sourced, representative dataset is supplied.

This roadmap separates completed functions from planned functions so that reports
and presentations never confuse demo values with reproducible results.

## Phase 1 — Resume parsing foundation (completed)

- [x] Define `ParsedDocument`, `IdentityInfo`, `CandidateProfile`,
      `JobRequirements`, and `ScoreResult`
- [x] Extract PDF, DOCX, and text content
- [x] Clean extracted text while preserving technical tokens
- [x] Recognize common resume sections
- [x] Extract identity into a separate data structure
- [x] Normalize skills with an editable taxonomy
- [x] Extract education and professional experience
- [x] Export structured JSON
- [x] Add automated tests and a sample resume
- [x] Extract LinkedIn, GitHub, portfolio, PDF, and DOCX hyperlinks

Acceptance criterion: a supported resume produces a structured candidate profile
with warnings for information that cannot be confidently extracted.

## Phase 2 — Job requirements and transparent matching (completed)

- [x] Parse job title and description
- [x] Separate mandatory, preferred, and unclassified relevant skills
- [x] Extract minimum experience, degree, field of study, and certifications
- [x] Calculate skill, experience, education, and certification coverage
- [x] Keep mandatory eligibility separate from ranking
- [x] Rank eligible candidates before ineligible candidates
- [x] Explain every match and failure

Acceptance criterion: one job description and several candidate profiles produce
an explainable baseline ranking without machine learning.

## Phase 3 — Text similarity and role classification (ML foundation completed)

- [x] Include a documented synthetic dataset for pipeline verification
- [ ] Replace the synthetic dataset with a sourced, representative real dataset
- [x] Remove duplicates and inspect class balance
- [x] Create train, validation, and test splits with a fixed random seed
- [x] Train TF-IDF with Logistic Regression, SVM, and Random Forest baselines
- [x] Add three-fold sigmoid calibration for Linear SVM probabilities
- [ ] Validate probability calibration quality on a real representative dataset
- [x] Evaluate accuracy, macro/weighted F1, and confusion matrix
- [x] Save the reproducible model pipeline and evidence artifacts

Acceptance criterion: every reported result can be regenerated from a documented
command, dataset version, and random seed.

## Phase 4 — Explainable composite scoring (completed)

- [x] Implement normalized TF-IDF text similarity
- [x] Use probability for the model-inferred job role rather than the candidate's top class
- [x] Combine similarity, role probability, and requirement score
- [x] Display component-level explanations
- [x] Preserve Phase 2 mandatory eligibility during hybrid ranking
- [x] Validate and normalize configurable component weights
- [x] Record exact per-component contributions and formula arithmetic
- [x] Support reviewed target-role overrides and low-confidence warnings
- [x] Test score boundaries, invalid weights, contribution totals, and role overrides

Acceptance criterion: the displayed fit percentage can be traced to its exact
inputs, weights, and calculations.

## Phase 5 — Growth-potential engine (completed)

- [x] Build a taxonomy-validated directed skill-prerequisite graph
- [x] Mark current and missing skill nodes
- [x] Estimate learning steps and rough planning hours
- [x] Simulate adding recommended skills and their prerequisites
- [x] Recalculate requirement coverage, potential fit, and eligibility
- [x] Keep text similarity and role probability fixed until a new résumé is supplied
- [ ] Complete external subject-matter review of every prerequisite relationship

Acceptance criterion: claims such as `72% -> 90%` are computed from the same
scoring engine rather than manually entered.

## Phase 6 — Fairness audit (completed)

- [x] Mask names, pronouns, contact details, and recognizable institution names carefully
- [x] Preserve degree and job-relevant qualifications
- [x] Rescore original and masked profiles
- [x] Measure normal score variation to choose a review threshold
- [x] Flag sensitivity for human review; never auto-reject

Acceptance criterion: the audit reports score sensitivity accurately and does not
claim to prove the absence of discrimination.

## Phase 7 — CSM216 algorithms and visualization

- [ ] Benchmark linear and binary search with repeated measurements
- [ ] Compare with Python `set` lookup for production context
- [ ] Build Matplotlib evaluation charts
- [ ] Build the Pygame skill-gap explorer as a separate process

Acceptance criterion: benchmark methodology, input sizes, repetition count, and
median timings are included with the results.

## Phase 8 — Tkinter application and final validation

- [ ] Create recruiter upload and ranking views
- [ ] Create candidate detail and explanation views
- [ ] Launch Pygame safely outside the Tkinter event loop
- [ ] Test malformed documents, scanned PDFs, and unseen resumes
- [ ] Replace all presentation mock numbers with verified results
- [ ] Add sources for external statistics

Acceptance criterion: a recruiter can choose a job description and resume folder,
review explainable results, open the skill explorer, and export a report.
