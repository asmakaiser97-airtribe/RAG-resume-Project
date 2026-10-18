# RAG-Based Profile Matching

A beginner-friendly retrieval-augmented resume matching project. It chunks resumes by section, creates sentence-transformer embeddings, stores chunks in ChromaDB, and ranks candidates using semantic similarity, skill overlap, and experience checks.

## Features
- Section-aware chunking and metadata extraction (candidate, skills, experience, education, path)
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Persistent ChromaDB vector collection
- Hybrid matching: semantic similarity + identified skill overlap
- Transparent 0–100 heuristic score, matched skills, excerpts, reasoning, must-have checks
- JSON output, 30 sample resumes, 5 sample job descriptions
- Basic experimentation notebook and retrieval evaluation scaffold

## Setup (Python 3.12 recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
The first run downloads the embedding model and can take a few minutes.

## Build the vector index
```powershell
python resume_rag.py
```

## Match a job description
```powershell
python job_matcher.py --job-file data/job_descriptions/python_backend.txt --allow-nonqualifiers --output output/python_backend_matches.json
```
By default, must-have requirements are enforced. Use `--allow-nonqualifiers` to see all candidates and compare them. If strict filtering returns fewer than 10 candidates, that means fewer candidates met all detected skill and experience conditions.

## Example JSON
```json
{
  "job_description": "Python backend developer with Django and PostgreSQL...",
  "top_matches": [
    {
      "candidate_name": "Example Candidate",
      "resume_path": "resumes/example_candidate.txt",
      "match_score": 83,
      "matched_skills": ["django", "postgresql", "python"],
      "relevant_excerpts": ["Experience ..."],
      "reasoning": "Semantic relevance 0.76; matched 3/4 identified job skills; candidate experience 4 years vs 3 required.",
      "must_have_pass": true
    }
  ]
}
```

## Design notes
- Chunking preserves common resume section headings where present. Long sections are split into bounded character windows.
- The embedding model is local; no paid LLM API key is required.
- Score formula: 45% semantic relevance + 40% detected job-skill coverage + 15% experience sufficiency. This is a demonstrative heuristic, not an objective measure of candidate quality.
- Must-have filter checks all detected skills in the job description plus explicit years threshold. In real hiring systems, distinguish truly mandatory skills from preferred skills instead of treating every detected skill as mandatory.
- Metadata extraction is deliberately lightweight and regex-based. It should be improved for varied real-world resume formats.
- Current reader supports TXT files. For PDF/DOCX, add parsers such as `pypdf` and `python-docx`; these are not silently treated as readable.
- Dataset resumes are synthetic demo data and must not be represented as real people.

## Evaluation
Open `experiments.ipynb` to run the sample experiment. Record precision@k/recall@k using a manually reviewed relevance label set and measure latency over repeated runs. Current latency is wall-clock time for one query, and the first query may include model initialization if the script was just started.

## Limitations
This is an educational prototype. Scores depend on keyword detection and metadata quality; they are not calibrated hiring predictions. Review excerpts and criteria before using results.
