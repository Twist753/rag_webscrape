"""
Improved embed_index.py
Now uses predefined tech_keywords and skill_list to extract structured skills
"""

import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from data.skills_data import tech_keywords, skill_list 

JSON_PATH = "data/assessments.json"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "shl_assessments"

def parse_duration_min(duration_str: str):
    if not duration_str:
        return None
    m = re.search(r"(\d{1,4})", str(duration_str))
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    return None


def extract_known_skills(text: str):
    """
    Extract skills and technologies from text by exact or partial matching.
    """
    text_lower = text.lower()
    matched = set()

    for kw in tech_keywords + skill_list:
        if kw.lower() in text_lower:
            matched.add(kw)

    return sorted(matched)


def make_augmented_text(a: dict):
    """
    Build a high-quality embedding text emphasizing extracted skills, test_type, and duration.
    """
    name = a.get("name", "")
    desc = a.get("description", "")
    test_type = a.get("test_type", "")
    duration = a.get("duration", "")

    # Extract known skills & technologies
    detected_skills = extract_known_skills(name + " " + desc)
    skills_text = ", ".join(detected_skills)

    # Fallback if nothing found
    if not detected_skills:
        skills_text = "General Aptitude, Communication, Problem Solving"

    minutes = parse_duration_min(duration)
    minutes_text = f"{minutes} minutes" if minutes else duration

    # Structured, weighted document text
    doc = f"""
    Assessment: {name}.
    Description: {desc}.
    Skills and Technologies: {skills_text}. {skills_text}. #weighted
    Test Type: {test_type}.
    Duration: {minutes_text}.
    """

    return doc.strip(), detected_skills, minutes


def main():
    if not os.path.exists(JSON_PATH):
        print(f" JSON not found: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        assessments = json.load(f)

    print(f"Loaded {len(assessments)} assessments.")

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=PERSIST_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print("Loaded existing collection.")
    except:
        collection = client.create_collection(name=COLLECTION_NAME)
        print("Created new collection.")

    if collection.count() > 0:
        print(f"⚠ Collection already has {collection.count()} items. Delete it if re-indexing.")
        return

    docs, metas, ids = [], [], []

    for idx, a in enumerate(assessments):
        doc_text, detected_skills, mins = make_augmented_text(a)
        docs.append(doc_text)
        metas.append({
            "name": a["name"],
            "url": a["url"],
            "test_type": a["test_type"],
            "duration_raw": a["duration"],
            "duration_mins": mins,
            "description": a["description"],
            "skills": ", ".join(detected_skills)
        })
        ids.append(f"assess_{idx}")

    print("Encoding embeddings...")
    embeddings = model.encode(docs, batch_size=128, show_progress_bar=True, convert_to_numpy=True)

    print("Sanitizing metadata before adding...")

    def sanitize_meta(meta):
        clean_meta = {}
        for k, v in meta.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, list):
                clean_meta[k] = ", ".join(map(str, v))
            elif isinstance(v, (dict, set)):
                clean_meta[k] = str(v)
            elif isinstance(v, (int, float, bool, str)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        return clean_meta

    metas = [sanitize_meta(m) for m in metas]

    print("Adding to ChromaDB...")
    collection.add(
        documents=docs,
        metadatas=metas,
        ids=ids,
        embeddings=embeddings.tolist()
    )


    print(f"✅ Indexed {len(docs)} assessments with enhanced skill tagging.")


if __name__ == "__main__":
    main()
