import os
import json
import re
from typing import List, Dict
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "shl_assessments"

load_dotenv()

class AssessmentRAG:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
            
                self.generation_config = genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
                self.llm = genai.GenerativeModel(
                    "gemini-2.5-flash", 
                    generation_config=self.generation_config
                )

                self.use_llm = True
                print("✅ Gemini initialized.")
            except Exception as e:
                print(f"⚠ Gemini init failed: {e}")
                self.use_llm = False
        else:
            print("⚠ GEMINI_API_KEY not found. LLM features disabled.")
            self.use_llm = False

        # Connect to Chroma
        try:
            self.client = chromadb.PersistentClient(path=PERSIST_DIR)
            self.collection = self.client.get_collection(COLLECTION_NAME)
            print("✅ Connected to Chroma collection:", COLLECTION_NAME)
        except Exception as e:
            print(f"❌ Failed to connect to ChromaDB: {e}")
            raise

    # ---------- LLM-based Query Parsing ----------
    def parse_query_with_llm(self, query: str) -> Dict:
        """Extract structured info from query. Falls back to heuristic parsing if fails."""
        if not query:
            return self._heuristic_parse(query)
        if not self.use_llm:
            return self._heuristic_parse(query)

        prompt = f"""
    Extract structured info from this hiring query.
    Return ONLY a valid JSON object with keys:
    - enhanced_query
    - required_skills (list)
    - test_types_needed (list from ["K","P","C"])
    - duration_constraint (integer minutes or null)

    Query: \"\"\"{query}\"\"\"
    """
        try:
            # --- THIS IS THE FIX (PART 2) ---
            resp = self.llm.generate_content(prompt)
            # The .text is now guaranteed to be a valid JSON string
            # No more regex or string splitting needed!
            txt = (resp.text or "{}").strip()
            data = json.loads(txt)
            # --- END OF FIX ---
        except Exception as e:
            # This will now only catch true API failures or if the
            # LLM fails to generate JSON at all (e.g., content safety)
            print(f"⚠ LLM parse failed: {e}") 
            print(f"Failed on text: {txt[:200]}") # Added for better debugging
            return self._heuristic_parse(query)

        # Clean and normalize
        data["enhanced_query"] = data.get("enhanced_query", query).strip()
        data["required_skills"] = [s.capitalize() for s in data.get("required_skills", [])]
        data["test_types_needed"] = [t for t in data.get("test_types_needed", []) if t in ["K", "P", "C"]]
        dur = data.get("duration_constraint")
        data["duration_constraint"] = int(dur) if isinstance(dur, (int, float, str)) and str(dur).isdigit() else None

        return data

    # ---------- Heuristic Parsing ----------
    def _heuristic_parse(self, query: str) -> Dict:
        skills = re.findall(
            r"\b(Java|Python|SQL|C\+\+|C#|Javascript|React|Teamwork|Communication|Leadership|Excel|Data|Cognitive)\b",
            query,
            flags=re.I,
        )
        skills = list({s.capitalize() for s in skills})
        ql = query.lower()

        test_types = []
        if any(w in ql for w in ["personality", "behaviour", "behavior", "fit", "culture", "communication", "leadership", "teamwork"]):
            test_types.append("P")
        if any(w in ql for w in ["skill", "knowledge", "technical", "developer", "python", "sql", "java"]):
            test_types.append("K")
        if any(w in ql for w in ["competency", "competencies", "cognitive"]):
            test_types.append("C")

        # Duration
        duration = None
        m = re.search(r"(\d{1,3})\s*(minutes|min|mins|hour|hr|h)", query, flags=re.I)
        if m:
            v = int(m.group(1))
            unit = m.group(2).lower()
            if "hour" in unit or "hr" in unit or "h" in unit:
                v *= 60
            duration = v

        enhanced = f"{query}. Relevant skills: {', '.join(skills)}."
        return {
            "enhanced_query": enhanced,
            "required_skills": skills,
            "test_types_needed": list(set(test_types)),
            "duration_constraint": duration,
        }

    # ---------- Metadata Filter ----------
    def build_where_filter(self, parsed: Dict) -> Dict:
        clauses = []
        test_types = parsed.get("test_types_needed", [])
        if test_types:
            test_types = [t.upper() for t in test_types if t in ["K", "P", "C"]]
            if len(test_types) == 1:
                clauses.append({"test_type": {"$eq": test_types[0]}})
            else:
                clauses.append({"test_type": {"$in": test_types}})
        duration_val = parsed.get("duration_constraint")
        if isinstance(duration_val, (int, float)) and duration_val > 0:
            clauses.append({"duration_mins": {"$lte": int(duration_val)}})
        if not clauses:
            return {}
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    # ---------- Retrieval ----------
    def retrieve(self, query: str, n_results: int = 40) -> (Dict, List[Dict]):
        parsed = self.parse_query_with_llm(query)
        search_text = parsed.get("enhanced_query") or query
        where = self.build_where_filter(parsed)

        try:
            collection_count = self.collection.count()
            if collection_count == 0:
                print("⚠ Collection empty.")
                return parsed, []
            n_results_safe = min(n_results, collection_count)

            query_kwargs = {"query_texts": [search_text], "n_results": n_results_safe}
            if where:
                query_kwargs["where"] = where

            results = self.collection.query(**query_kwargs)

        except Exception as e:
            print(f"⚠ Query error {e}, retrying without filter...")
            try:
                results = self.collection.query(query_texts=[search_text], n_results=10)
            except Exception as e2:
                print(f"❌ Chroma query failed: {e2}")
                return parsed, []

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [0] * len(docs)

        candidates = []
        for i, d in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            candidates.append({
                "name": meta.get("name"),
                "url": meta.get("url"),
                "test_type": meta.get("test_type"),
                "duration_mins": meta.get("duration_mins"),
                "description": meta.get("description"),
                "skills": meta.get("skills", []),
                "score": round(1 - distances[i], 4) if distances[i] is not None else 0
            })
        return parsed, candidates

    # ----------- Recommend ------------
    def recommend(self, query: str, top_k: int = 10, min_k: int = 5) -> List[Dict]:
        parsed, candidates = self.retrieve(query, n_results=60)
        if not candidates:
            return []
    
        # Sort once by score (desc)
        candidates = sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)
    
        # --- Adaptive cutoff ---
        # Use the top 20 to estimate a good threshold.
        top_pool = candidates[:20]
        if not top_pool:
            return []
    
        # Score at rank 10 (or last if fewer) minus a small margin,
        # but never below a floor of 0.55
        rank_for_cut = min(9, len(top_pool) - 1)
        score_at_rank = top_pool[rank_for_cut].get("score", 0)
        adaptive_cutoff = max(0.55, score_at_rank - 0.05)
    
        filtered = [c for c in candidates if c.get("score", 0) >= adaptive_cutoff]
    
        # Ensure at least min_k
        if len(filtered) < min_k:
            filtered = candidates[:min_k]
    
        # Cap at top_k (this already biases toward ~10 if quality allows)
        shortlisted = filtered[:top_k]
    
        # --- Balance across requested test types if present ---
        requested_types = parsed.get("test_types_needed") or []
        requested_types = [t for t in requested_types if t in ["K", "P", "C"]]
    
        if requested_types:
            by_type = {"K": [], "P": [], "C": []}
            for c in shortlisted:
                t = (c.get("test_type") or "").upper()
                by_type[t if t in by_type else "other"].append(c)
    
            balanced = []
            # Round-robin pull respecting requested types first, then fill with others
            while len(balanced) < min(top_k, len(shortlisted)):
                any_added = False
                for t in requested_types:
                    if by_type[t]:
                        balanced.append(by_type[t].pop(0))
                        any_added = True
                        if len(balanced) >= top_k:
                            break
                if len(balanced) >= top_k:
                    break
                if not any_added:
                    # Fill from remaining buckets
                    for t in ["K", "P", "C", "other"]:
                        if by_type[t]:
                            balanced.append(by_type[t].pop(0))
                            any_added = True
                            if len(balanced) >= top_k:
                                break
                if not any_added:
                    break
                
            shortlisted = balanced
    
        return shortlisted[:top_k]

    # ---------- Format for Web Frontend ----------

    def format_for_web(self, recommendations: List[Dict]) -> List[Dict]:
        """Format output cleanly for API/Streamlit frontend."""
        formatted = []
        for r in recommendations:
            skills_data = r.get("skills", [])
            
            if isinstance(skills_data, str):
                if "," in skills_data:
                    skills_data = [s.strip() for s in skills_data.split(",") if s.strip()]
                else:
                    skills_data = [skills_data.strip()] if skills_data.strip() else []
            elif not isinstance(skills_data, list):
                skills_data = []

            formatted.append({
                "Assessment Name": r.get("name"),
                "Type": r.get("test_type"),
                "Duration (mins)": r.get("duration_mins"),
                "Skills": ", ".join(skills_data),
                "Description": (r.get("description") or "")[:200] + "...",
                "URL": r.get("url")
            })
        return formatted

# ---------- CLI Test ----------
if __name__ == "__main__":
    rag = AssessmentRAG()
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for assessments that can be completed in 40 minutes.",
        "Need someone with strong communication, leadership and cultural fit for a senior manager role. Duration ~60 minutes.",
        "Hiring for Python + SQL mid-level analyst, test should be 30-40 mins and include cognitive + personality checks."
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print("QUERY:", q)
        recs = rag.recommend(q, top_k=10)
        formatted = rag.format_for_web(recs)

        print(f"Returned {len(formatted)} recommendations:\n")
        for i, item in enumerate(formatted, 1):
            print(f"{i}. {item['Assessment Name']} ({item['Type']}) - {item['Duration (mins)']} mins")
            print(f"   Skills: {item['Skills']}")
            print(f"   URL: {item['URL']}\n")
        print("-" * 60)
