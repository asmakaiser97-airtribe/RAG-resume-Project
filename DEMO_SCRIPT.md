# 3–4 minute demo video script

1. (0:00–0:25) Introduce the problem: matching resumes to a job description.
2. (0:25–0:55) Show the architecture: section-aware chunks → embeddings → ChromaDB → hybrid ranking.
3. (0:55–1:25) Run `python resume_rag.py` and show the indexed resume/chunk counts.
4. (1:25–2:20) Run `python job_matcher.py --job-file data/job_descriptions/python_backend.txt --allow-nonqualifiers` and explain score, matched skills, excerpts, and reasoning.
5. (2:20–3:00) Show the JSON output and one alternate JD, such as `ml_engineer.txt`.
6. (3:00–3:30) Show `experiments.ipynb`, explain precision@k/recall@k and query latency.
7. (3:30–3:45) State limitations: synthetic dataset, TXT parser, heuristic score, and need for human review.
