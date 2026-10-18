"""
Job matching engine: semantic retrieval + critical skill keyword checks + transparent scoring.
Run: python job_matcher.py --job-file data/job_descriptions/python_backend.txt
"""
import argparse, json, re, time
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from resume_rag import BASE_DIR, DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL, build_index, get_collection

SKILL_ALIASES = {
    "django rest framework": ["django rest framework", "drf"],
    "postgresql": ["postgresql", "postgres"],
    "rest api": ["rest api", "rest apis", "restful api", "restful apis"],
    "machine learning": ["machine learning", "ml"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "fastapi": ["fastapi"],
    "python": ["python"],
    "django": ["django"],
    "docker": ["docker"],
    "redis": ["redis"],
    "celery": ["celery"],
    "aws": ["aws", "amazon web services"],
    "kubernetes": ["kubernetes", "k8s"],
    "pytorch": ["pytorch"],
    "embeddings": ["embeddings", "embedding"],
    "rag": ["retrieval-augmented generation", "rag"],
}
_MODEL = None

def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL

def norm(text): return re.sub(r"[^a-z0-9+#. ]+", " ", text.lower())

def extract_skills(text):
    t = norm(text)
    found = []
    for skill, aliases in SKILL_ALIASES.items():
        if any(re.search(r"(?<!\w)" + re.escape(norm(a).strip()) + r"(?!\w)", t) for a in aliases):
            found.append(skill)
    return found

def extract_must_have_skills(text):
    match = re.search(
        r"must[- ]have\s+skills?\s*:\s*(.*?)(?:\n|$)",
        text,
        re.I,
    )

    if match:
        return extract_skills(match.group(1))

    match = re.search(
        r"required\s+skills?\s*:\s*(.*?)(?:\n|$)",
        text,
        re.I,
    )

    if match:
        return extract_skills(match.group(1))

    return extract_skills(text)

def required_years(text):
    patterns = [
        r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?"
        r"(?:professional\s+)?(?:relevant\s+)?experience",
        r"at least\s+(\d+)\s+years?",
        r"minimum\s+of\s+(\d+)\s+years?",
        r"(\d+)\s*\+?\s*years?\s+of\s+Python",
        r"(\d+)\s*\+?\s*years?\s+of\s+backend",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m: return int(m.group(1))
    return 0

def match_job(job_description: str, top_k: int = 10, enforce_must_haves: bool = True):
    start = time.perf_counter()
    collection = get_collection()
    if collection.count() == 0:
        build_index()
        collection = get_collection()
    job_skills = extract_skills(job_description)
    must_have_skills = extract_must_have_skills(job_description)
    years_required = required_years(job_description)
    model = get_embedding_model()
    q_emb = model.encode([job_description], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=min(max(top_k * 5, top_k), collection.count()),
                              include=["documents", "metadatas", "distances"])
    grouped = {}
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        key = meta["resume_path"]
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        if key not in grouped:
            grouped[key] = {"candidate_name": meta["candidate_name"], "resume_path": meta["resume_path"],
                            "experience_years": int(meta.get("experience_years", 0)),
                            "skills_text": meta.get("skills_text", ""), "semantic": similarity,
                            "excerpts": [], "sections": []}
        else:
            grouped[key]["semantic"] = max(grouped[key]["semantic"], similarity)
        if len(grouped[key]["excerpts"]) < 3:
            grouped[key]["excerpts"].append(doc[:400])
            grouped[key]["sections"].append(meta.get("section", "Profile"))
    candidates = []
    for item in grouped.values():
        candidate_skills = set(extract_skills(item["skills_text"] + " " + " ".join(item["excerpts"])))
        matched = sorted(candidate_skills.intersection(job_skills))
        skill_coverage = len(matched) / max(1, len(job_skills))
        years = item["experience_years"]
        experience_score = 1.0 if years_required == 0 else min(years / years_required, 1.0)
        # Explainable weighted score: semantic 45%, skill coverage 40%, experience 15%.
        score = round(100 * (0.45 * item["semantic"] + 0.40 * skill_coverage + 0.15 * experience_score))
        missing = sorted(set(job_skills) - candidate_skills)
        missing_must_haves = sorted(
            set(must_have_skills) - candidate_skills
        )

        meets_years = years >= years_required
        meets_skills = len(missing_must_haves) == 0
        passes = meets_years and meets_skills
        reasoning = (f"Semantic relevance {item['semantic']:.2f}; matched {len(matched)}/{len(job_skills)} "
                     f"identified job skills; candidate experience {years} years"
                     + (f" vs {years_required} required." if years_required else " (no explicit year threshold)."))
        if missing: reasoning += " Missing identified skills: " + ", ".join(missing) + "."
        if not meets_years: reasoning += " Does not meet the stated experience threshold."
        item_out = {
            "candidate_name": item["candidate_name"], "resume_path": item["resume_path"],
            "match_score": score, "matched_skills": matched,
            "relevant_excerpts": item["excerpts"],
            "reasoning": reasoning, "experience_years": years,
            "must_have_pass": passes, "missing_skills": missing,
            "must_have_skills": must_have_skills,
            "missing_must_have_skills": missing_must_haves,
            "matched_sections": sorted(set(item["sections"]))
        }
        if not enforce_must_haves or passes:
            candidates.append(item_out)
    candidates.sort(key=lambda x: (x["match_score"], x["experience_years"]), reverse=True)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    return {"job_description": job_description, "top_matches": candidates[:top_k],
            "filters": {"required_experience_years": years_required
                        , "identified_job_skills": job_skills,
                        "must_have_skills": must_have_skills,
                        "enforce_must_haves": enforce_must_haves},
            "metrics": {"latency_ms": elapsed_ms, "candidates_returned": min(top_k, len(candidates))}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", type=str, help="Path to a job description text file")
    parser.add_argument("--job", type=str, help="Job description passed directly")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-nonqualifiers", action="store_true", help="Return candidates even if must-haves fail")
    parser.add_argument("--output", type=str, help="Optional JSON output path")
    args = parser.parse_args()
    if args.job_file:
        path = Path(args.job_file)
        if not path.is_absolute(): path = BASE_DIR / path
        job = path.read_text(encoding="utf-8")
    elif args.job:
        job = args.job
    else:
        job = input("Paste job description, then press Enter:\n")
    result = match_job(job, top_k=args.top_k, enforce_must_haves=not args.allow_nonqualifiers)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute(): out_path = BASE_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\nSaved JSON to {out_path}")

if __name__ == "__main__":
    main()
