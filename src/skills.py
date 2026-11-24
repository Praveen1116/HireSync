import re
import json
from pathlib import Path

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


def analyze_skill_gap(resume_text: str, jd_text: str) -> dict:
    """
    Performs skill gap analysis between resume and job description.
    
    Returns:
        dict containing:
        - resume_skills: list of skills in resume
        - jd_skills: list of skills in JD
        - matched_skills: skills present in both (for green highlights)
        - missing_skills: JD skills absent in resume (for red/gap highlights)
        - overlap_ratio: fraction of JD skills found in resume (0.0 to 1.0)
    """
    resume_skills = extract_skills_from_text(resume_text)
    jd_skills = extract_skills_from_text(jd_text)

    matched = resume_skills.intersection(jd_skills)
    missing = jd_skills.difference(resume_skills)

    overlap_ratio = len(matched) / len(jd_skills) if len(jd_skills) > 0 else 0.0

    return {
        "resume_skills": sorted(list(resume_skills)),
        "jd_skills": sorted(list(jd_skills)),
        "matched_skills": sorted(list(matched)),
        "missing_skills": sorted(list(missing)),
        "overlap_ratio": float(overlap_ratio)
    }
