"""
RAG Engine for Assessment Recommendations
Uses ChromaDB and sentence transformers for semantic search
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict
import os

class AssessmentRAG:
    def __init__(self):
        # Initialize embedding model
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection("shl_assessments")
            print("Loaded existing ChromaDB collection")
        except:
            self.collection = self.client.create_collection(
                name="shl_assessments",
                metadata={"hnsw:space": "cosine"}
            )
            print("Created new ChromaDB collection")
        
        self.assessments_data = []
    
    def load_and_index_assessments(self, json_path: str = "data/assessments.json"):
        """Load assessments from JSON and index in ChromaDB"""
        print(f"Loading assessments from {json_path}...")
        
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found. Run scraper.py first!")
            return
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.assessments_data = json.load(f)
        
        print(f"Loaded {len(self.assessments_data)} assessments")
        
        # Check if already indexed
        if self.collection.count() > 0:
            print(f"Collection already has {self.collection.count()} documents")
            return
        
        # Prepare data for indexing
        documents = []
        metadatas = []
        ids = []
        
        for idx, assessment in enumerate(self.assessments_data):
            # Create rich text representation
            text = f"{assessment['name']}. {assessment['description']}. Test Type: {assessment['test_type']}. Duration: {assessment['duration']}"
            
            documents.append(text)
            metadatas.append({
                "name": assessment['name'],
                "url": assessment['url'],
                "test_type": assessment['test_type'],
                "duration": assessment['duration']
            })
            ids.append(f"assess_{idx}")
        
        # Add to ChromaDB
        print("Indexing assessments in ChromaDB...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully indexed {len(documents)} assessments")
    
    def recommend(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Get recommendations for a query
        Returns list of recommended assessments with name and URL
        """
        print(f"\nQuery: {query}")
        
        # Search ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=min(20, self.collection.count())  # Get more candidates for re-ranking
        )
        
        if not results['documents'][0]:
            return []
        
        # Extract results
        candidates = []
        for idx in range(len(results['documents'][0])):
            candidates.append({
                "name": results['metadatas'][0][idx]['name'],
                "url": results['metadatas'][0][idx]['url'],
                "test_type": results['metadatas'][0][idx]['test_type'],
                "distance": results['distances'][0][idx] if 'distances' in results else 0
            })
        
        # Apply domain balancing logic
        balanced_results = self.balance_recommendations(query, candidates, top_k)
        
        return balanced_results
    
    def balance_recommendations(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Balance recommendations across different test types based on query
        """
        query_lower = query.lower()
        
        # Detect if query mentions multiple domains
        has_technical = any(word in query_lower for word in ['java', 'python', 'sql', 'technical', 'coding', 'programming', 'developer'])
        has_behavioral = any(word in query_lower for word in ['collaborate', 'personality', 'behavior', 'communication', 'teamwork', 'leadership'])
        has_cognitive = any(word in query_lower for word in ['cognitive', 'analytical', 'problem-solving', 'reasoning'])
        
        # If multiple domains detected, ensure balanced results
        if (has_technical and has_behavioral) or (has_technical and has_cognitive) or (has_behavioral and has_cognitive):
            print("Detected multi-domain query - applying balanced recommendations")
            
            # Separate by test type
            k_tests = [c for c in candidates if c['test_type'] == 'K']
            p_tests = [c for c in candidates if c['test_type'] == 'P']
            c_tests = [c for c in candidates if c['test_type'] == 'C']
            
            # Balance the results
            balanced = []
            target_per_type = max(2, top_k // 3)
            
            balanced.extend(k_tests[:target_per_type])
            balanced.extend(p_tests[:target_per_type])
            balanced.extend(c_tests[:target_per_type])
            
            # Fill remaining slots with best candidates
            remaining = [c for c in candidates if c not in balanced]
            balanced.extend(remaining[:top_k - len(balanced)])
            
            return balanced[:top_k]
        
        # Single domain query - return top candidates
        return candidates[:top_k]
    
    def evaluate_recall(self, test_queries: List[Dict]) -> float:
        """
        Calculate Mean Recall@10 for evaluation
        test_queries format: [{"query": "...", "relevant_urls": [...]}]
        """
        total_recall = 0
        
        for item in test_queries:
            query = item['query']
            relevant_urls = set(item['relevant_urls'])
            
            # Get recommendations
            recommendations = self.recommend(query, top_k=10)
            recommended_urls = set([r['url'] for r in recommendations])
            
            # Calculate recall for this query
            if len(relevant_urls) > 0:
                recall = len(recommended_urls & relevant_urls) / len(relevant_urls)
            else:
                recall = 0
            
            total_recall += recall
            print(f"Query: {query[:50]}... | Recall@10: {recall:.3f}")
        
        mean_recall = total_recall / len(test_queries)
        print(f"\nMean Recall@10: {mean_recall:.3f}")
        return mean_recall

def main():
    # Initialize RAG engine
    rag = AssessmentRAG()
    
    # Load and index assessments
    rag.load_and_index_assessments()
    
    # Test with sample query
    query = "I am hiring for Java developers who can also collaborate effectively with my business teams."
    recommendations = rag.recommend(query, top_k=5)
    
    print("\nRecommendations:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"{idx}. {rec['name']}")
        print(f"   URL: {rec['url']}")
        print(f"   Type: {rec['test_type']}\n")

if __name__ == "__main__":
    main()