"""TF-IDF cosine similarity for a resume and job description."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def text_similarity(resume_text: str, job_text: str) -> float:
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    try:
        matrix = vectorizer.fit_transform([resume_text, job_text])
    except ValueError:
        return 0.0
    score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    return round(max(0.0, min(1.0, score)), 6)
