import re
import json
from pathlib import Path
import numpy as np

# Path to the exported skills taxonomy
TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "models" / "skills_taxonomy.json"


def load_skills_taxonomy(taxonomy_path=TAXONOMY_PATH) -> set:
    """Loads all skills from the taxonomy JSON and returns a flat set of lowercase skills."""
    if not taxonomy_path.exists():
        # Fallback basic list if file is missing
        return {"python", "java", "sql", "aws", "docker", "machine learning", "react"}

    with open(taxonomy_path, "r") as f:
        taxonomy_data = json.load(f)

    all_skills = set()
    for category, skill_list in taxonomy_data.items():
        all_skills.update([s.lower().strip() for s in skill_list])
    return all_skills


# Global preloaded skill set for fast matching
SKILLS_SET = load_skills_taxonomy()


def extract_skills_from_text(text: str) -> set:
    """
    Extracts all skills from text using word-boundary matching.
    Case-insensitive and correctly recognizes symbols like c++, c#, .js, /
    """
    if not text or not isinstance(text, str):
        return set()

    text_lower = text.lower()
    found_skills = set()
    for skill in SKILLS_SET:
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills


def analyze_skill_gap(resume_text: str, jd_text: str, sbert_model=None, semantic_threshold: float = 0.75) -> dict:
    """
    Performs skill gap analysis between resume and job description.
    
    Features:
    - Layer 1: Exact keyword boundary matching from curated taxonomy.
    - Layer 2 (Method 2): Global SBERT Semantic Soft-Matching (sim >= 0.75)
      to recognize morphological forms (e.g., 'tests' matching 'testing',
      'managing' matching 'management') without hardcoding.
    
    Returns:
        dict containing matched_skills, missing_skills, and overlap_ratio.
    """
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    # 1. Exact string matches
    matched = set(resume_skills.intersection(jd_skills))
    unmatched_jd_skills = jd_skills.difference(matched)

    # 2. Global Semantic Soft-Matching using SBERT embeddings
    if sbert_model is not None and unmatched_jd_skills:
        resume_tokens = list(set(re.findall(r'[a-zA-Z\+\#\.\/]{3,}', resume_text.lower())))
        if resume_tokens:
            token_embs = sbert_model.encode(resume_tokens, convert_to_numpy=True, normalize_embeddings=True)
            for skill in list(unmatched_jd_skills):
                sk_emb = sbert_model.encode(skill, convert_to_numpy=True, normalize_embeddings=True)
                sims = np.dot(token_embs, sk_emb)
                if float(np.max(sims)) >= semantic_threshold:
                    matched.add(skill)

    missing = jd_skills.difference(matched)
    overlap_ratio = len(matched) / len(jd_skills) if len(jd_skills) > 0 else 0.0

    return {
        "resume_skills": sorted(list(resume_skills)),
        "jd_skills": sorted(list(jd_skills)),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "overlap_ratio": float(overlap_ratio)
    }
