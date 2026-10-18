"""
RAG-Based Profile Matching: resume ingestion, section-aware chunking, metadata, embeddings, ChromaDB.
Run: python resume_rag.py
"""
import re
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
RESUME_DIR = BASE_DIR / "resumes"
DB_DIR = BASE_DIR / "vector_db"
COLLECTION_NAME = "resume_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SECTION_NAMES = {"professional summary", "summary", "skills", "technical skills",
                 "experience", "work experience", "education", "certifications", "projects"}

def read_resume(path: Path) -> str:
    """Read TXT resumes. Extend with pypdf/python-docx for PDF/DOCX if needed."""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""

def extract_metadata(text: str, path: Path) -> Dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name = lines[0] if lines else path.stem.replace("_", " ").title()
    skills = []
    match = re.search(r"(?ims)^(?:technical )?skills\s*\n(.*?)(?=^\w[^\n]*\n|\Z)", text)
    if match:
        skills = [s.strip() for s in re.split(r",|;|\|", match.group(1).replace("\n", ","))
                  if s.strip()]
    exp_match = re.search(r"(?i)(\d+)\+?\s+years?\s+of\s+experience", text)
    if not exp_match:
        exp_match = re.search(r"(?i)(\d+)\s+years?", text)
    experience_years = int(exp_match.group(1)) if exp_match else 0
    edu_match = re.search(r"(?im)^education\s*\n(.*?)(?=^\w[^\n]*\n|\Z)", text)
    education = edu_match.group(1).strip().replace("\n", " ") if edu_match else "Not specified"
    return {"candidate_name": name, "skills": skills, "experience_years": experience_years,
            "education": education, "resume_path": str(path.relative_to(BASE_DIR)).replace("\\", "/")}

def section_aware_chunks(text: str, max_chars: int = 700) -> List[Dict[str, str]]:
    """Split by recognizable headings first, then split long sections into bounded chunks."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    chunks, current_heading, current = [], "Profile", []
    def flush():
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            # Further split long section text without cutting at arbitrary tiny lengths.
            for start in range(0, len(body), max_chars):
                piece = body[start:start + max_chars].strip()
                if piece:
                    chunks.append({"section": current_heading, "text": piece})
        current = []
    for line in lines:
        normalized = line.lower().rstrip(":")
        if normalized in SECTION_NAMES:
            flush()
            current_heading = line.rstrip(":")
        else:
            current.append(line)
    flush()
    return chunks

def build_index(reset: bool = True) -> Dict[str, int]:
    if not RESUME_DIR.exists():
        raise FileNotFoundError(f"Resume directory not found: {RESUME_DIR}")
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    model = SentenceTransformer(EMBEDDING_MODEL)
    ids, docs, metadatas, embeddings = [], [], [], []
    candidate_count = 0
    for path in sorted(RESUME_DIR.glob("*")):
        if path.suffix.lower() not in {".txt"}:
            continue
        text = read_resume(path)
        if not text.strip():
            continue
        meta = extract_metadata(text, path)
        chunks = section_aware_chunks(text)
        if not chunks:
            continue
        candidate_count += 1
        for idx, chunk in enumerate(chunks):
            ids.append(f"{path.stem}__{idx}")
            docs.append(chunk["text"])
            metadatas.append({
                "candidate_name": meta["candidate_name"],
                "resume_path": meta["resume_path"],
                "experience_years": int(meta["experience_years"]),
                "education": meta["education"][:500],
                "skills_text": ", ".join(meta["skills"])[:1000],
                "section": chunk["section"],
            })
    if docs:
        embeddings = model.encode(docs, normalize_embeddings=True).tolist()
        # Chroma upsert makes repeated runs safe.
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    return {"resumes_indexed": candidate_count, "chunks_indexed": len(docs)}

def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

if __name__ == "__main__":
    result = build_index()
    print("Resume index ready:", result)
