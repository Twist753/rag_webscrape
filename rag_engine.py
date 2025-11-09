"""
rag_service.py
- Loads ChromaDB created by embed_index.py
- For an input query:
    1. Uses Gemini LLM to parse the query into required_skills, test_types_needed, duration_constraint
    2. Constructs a metadata filter and an enhanced query text
    3. Queries ChromaDB for top candidates (vector search)
    4. Optionally re-ranks candidates via Gemini to ensure balance and final selection
    5. Returns top-K recommendations
"""

import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
import google.generativeai as genai

# Chroma config - must match embed_index.py
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "shl_assessments"

# load env
load_dotenv()

class AssessmentRAG:
    def __init__(self):
        # Gemini LLM init
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            try:
                self.llm = genai.GenerativeModel("gemini-2.5-flash")
                self.use_llm = True
                print("✅ Gemini initialized.")
            except Exception as e:
                print("⚠ Gemini init failed:", e)
                self.use_llm = False
        else:
            print("⚠ GEMINI_API_KEY not found. LLM features disabled.")
            self.use_llm = False

        # connect to chroma
        self.client = chromadb.Client(Settings(persist_directory=PERSIST_DIR, anonymized_telemetry=False))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print("✅ Connected to Chroma collection:", COLLECTION_NAME)

    # ---------- LLM parsing/enhancement ----------
    def parse_query_with_llm(self, query: str) -> Dict:
        """
        Ask LLM to parse the query into structured components.
        If LLM not available, return simple heuristics.
        """
        if not self.use_llm:
            # fallback: simple heuristics
            skills = re.findall(r"\b(Java|Python|SQL|C\+\+|C#|JavaScript|React|Communication|Leadership)\b", query, flags=re.I)
            skills = list({s.capitalize() for s in skills})
            test_types = []
            if any(w in query.lower() for w in ["personality", "behaviour", "behavior", "fit", "culture"]):
                test_types.append("P")
            if any(w in query.lower() for w in ["skill", "knowledge", "technical", "java", "python", "sql"]):
                test_types.append("K")
            duration = None
            m = re.search(r"(\d{1,3})\s*(minutes|min|mins|hour|hr|h)", query, flags=re.I)
            if m:
                v = int(m.group(1))
                unit = m.group(2).lower()
                if "hour" in unit or "hr" in unit or "h" in unit:
                    v = v * 60
                duration = v
            return {
                "enhanced_query": query,
                "required_skills": skills,
                "test_types_needed": test_types,
                "duration_constraint": duration
            }

        prompt = f"""
You are an assistant that extracts hiring intent from a short query.
Return ONLY a JSON object (no explanation) with keys:
- enhanced_query: an expanded version that includes synonyms/phrases useful for semantic search
- required_skills: list of important skills (strings)
- test_types_needed: list of test types among ["K","P","C"] (may be empty)
- duration_constraint: integer minutes maximum (or null if not specified)

Query:
\"\"\"{query}\"\"\"
"""
        try:
            resp = self.llm.generate_content(prompt)
            txt = resp.text.strip()
            # clean fences if present
            if txt.startswith("```"):
                txt = "\n".join(txt.split("```")[1:]).strip()
            data = json.loads(txt)
            return data
        except Exception as e:
            print("⚠ LLM parse failed:", e)
            # fallback
            return self.parse_query_with_llm(None and query or query)  # call heuristic fallback

    # ---------- build metadata filter ----------
    def build_where_filter(self, parsed: Dict):
        """
        Build a where (metadata) filter for chroma query. Chroma supports basic dict filters;
        here we'll produce a simple filter: test_type contains any requested type and
        duration_mins <= duration_constraint (if available).
        Note: Adjust this depending on your Chroma version's 'where' syntax.
        """
        where = {}
        # Filter by test type if provided
        t_needed = parsed.get("test_types_needed") or []
        if t_needed:
            # We'll store a filter that checks that the metadata.test_type string contains at least
            # one of the requested characters. Chroma's python client supports "where" with simple equality dicts
            # For broader compatibility we'll do a list of OR conditions by using metadata fields.
            # Many chroma versions allow {"$or": [{"test_type": {"$contains": "K"}}, ...]}
            ors = []
            for t in t_needed:
                # Use contains match
                ors.append({"test_type": {"$contains": t}})
            where["$or"] = ors

        # duration constraint
        dur = parsed.get("duration_constraint")
        if dur is not None:
            # Keep items whose metadata.duration_mins is None or <= dur.
            # Chroma's 'where' cannot express OR easily across None; we'll simply filter post-query if needed.
            where["duration_mins"] = {"$lte": dur}
        return where

    # ---------- retrieval ----------
    def retrieve(self, query: str, n_results: int = 20) -> List[Dict]:
        parsed = self.parse_query_with_llm(query)
        search_text = parsed.get("enhanced_query") or query

        where = self.build_where_filter(parsed)

        # perform vector search
        try:
            # If where is empty, avoid sending empty filter that some chroma versions reject
            query_kwargs = {"query_texts": [search_text], "n_results": min(n_results, self.collection.count())}
            if where:
                query_kwargs["where"] = where

            results = self.collection.query(**query_kwargs)
        except Exception as e:
            print("⚠ Chroma query error (trying without where filter):", e)
            # fallback to no where
            results = self.collection.query(query_texts=[search_text], n_results=min(n_results, self.collection.count()))

        # parse results
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [0]*len(docs)

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
                "score": 1 - distances[i] if distances else None
            })
        return parsed, candidates

    # ---------- LLM re-ranking to ensure balance ----------
    def rerank_with_llm(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        if not self.use_llm or not candidates:
            return candidates[:top_k]

        # Prepare a short candidate list (max 15)
        cand_texts = []
        for i, c in enumerate(candidates[:20], 1):
            cand_texts.append(f"{i}. {c.get('name')} | Type:{c.get('test_type')} | Duration:{c.get('duration_mins')}\n   { (c.get('description') or '')[:150]}")

        prompt = f"""
You are an expert HR recommender. The user's query is:
\"\"\"{query}\"\"\"

From the numbered list below, pick the {top_k} assessments that best match the query.
- Ensure balance: if the query requests both technical (K) and personality (P) aspects, include both.
- Consider skills and duration.

List:
{chr(10).join(cand_texts)}

Return ONLY a JSON array of integers (1-based indices), e.g. [1,4,2]
"""
        try:
            resp = self.llm.generate_content(prompt)
            txt = resp.text.strip()
            # unwrap fences
            if txt.startswith("```"):
                txt = "\n".join(txt.split("```")[1:]).strip()
            selected = json.loads(txt)
            # convert to zero-based indices
            idxs = [int(i)-1 for i in selected if isinstance(i, int) or (isinstance(i, str) and i.isdigit())]
            reranked = [candidates[i] for i in idxs if 0 <= i < len(candidates)]
            return reranked[:top_k]
        except Exception as e:
            print("⚠ LLM re-rank failed:", e)
            return candidates[:top_k]

    # ---------- public recommend ----------
    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        parsed, candidates = self.retrieve(query, n_results=40)
        if not candidates:
            return []

        # Optional post-filter to ensure balance if parsed requests multiple types:
        requested_types = parsed.get("test_types_needed") or []
        if requested_types:
            # make sure we return a balanced mix if possible
            by_type = {"K": [], "P": [], "C": [], "other": []}
            for c in candidates:
                t = c.get("test_type") or ""
                placed = False
                for key in ["K","P","C"]:
                    if key in (t or ""):
                        by_type[key].append(c)
                        placed = True
                        break
                if not placed:
                    by_type["other"].append(c)
            # simple balancing: interleave from requested types then fill with others
            balanced = []
            # cycle through requested types and pop one from each until we have enough
            i = 0
            while len(balanced) < top_k:
                any_added = False
                for t in requested_types:
                    if by_type.get(t):
                        balanced.append(by_type[t].pop(0))
                        any_added = True
                        if len(balanced) >= top_k:
                            break
                if not any_added:
                    # take from other lists
                    for k in by_type:
                        if by_type[k]:
                            balanced.append(by_type[k].pop(0))
                            any_added = True
                            if len(balanced) >= top_k:
                                break
                if not any_added:
                    break
            candidates = balanced

        # Re-rank with LLM for final top_k if available
        recommendations = self.rerank_with_llm(query, candidates, top_k=top_k) if self.use_llm else candidates[:top_k]

        return recommendations

# ---------- simple CLI test ----------
if __name__ == "__main__":
    rag = AssessmentRAG()
    test_queries = [
        "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
        "Need someone with strong communication, leadership and cultural fit for a senior manager role. Duration ~60 minutes.",
        "Hiring for Python + SQL mid-level analyst, test should be 30-40 mins and include cognitive + personality checks."
    ]
    for q in test_queries:
        print("\n" + "="*60)
        print("QUERY:", q)
        recs = rag.recommend(q, top_k=6)
        print(f"Returned {len(recs)} recommendations:")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r['name']} | Type:{r.get('test_type')} | Duration:{r.get('duration_mins')} mins | {r.get('url')}")
        print("-"*60)
