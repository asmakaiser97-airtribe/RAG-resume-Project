"""Basic smoke test after installing requirements and building the index."""
from pathlib import Path
from resume_rag import build_index
from job_matcher import match_job

def test_index_and_matching():
    stats = build_index(reset=True)
    assert stats["resumes_indexed"] >= 30
    job = Path("data/job_descriptions/python_backend.txt").read_text(encoding="utf-8")
    result = match_job(job, top_k=10, enforce_must_haves=False)
    assert result["job_description"]
    assert isinstance(result["top_matches"], list)
    assert len(result["top_matches"]) > 0
    assert all(0 <= x["match_score"] <= 100 for x in result["top_matches"])
