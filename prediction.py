"""
Evaluation Script for RAG System
Uses labeled train set to calculate Mean Recall@10
"""

import pandas as pd
from rag_engine import AssessmentRAG
import os

class RAGEvaluator:
    def __init__(self):
        self.rag = AssessmentRAG()
        self.rag.load_and_index_assessments()
    
    def load_train_data(self, csv_path: str = "data/train_labeled.csv"):
        """
        Load labeled training data
        Expected format: Query, Assessment_url (multiple rows per query)
        """
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found!")
            return []
        
        df = pd.read_csv(csv_path)
        
        # Group by query to get all relevant URLs per query
        queries_dict = {}
        for _, row in df.iterrows():
            query = row['Query']
            url = row['Assessment_url']
            
            if query not in queries_dict:
                queries_dict[query] = []
            queries_dict[query].append(url)
        
        # Convert to list format
        train_data = [
            {"query": query, "relevant_urls": urls}
            for query, urls in queries_dict.items()
        ]
        
        print(f"Loaded {len(train_data)} unique queries from train set")
        return train_data
    
    def evaluate_on_train(self):
        """
        Evaluate RAG system on labeled train set
        Calculate Mean Recall@10
        """
        train_data = self.load_train_data()
        
        if not train_data:
            print("No train data found. Skipping evaluation.")
            return
        
        print("\n" + "="*60)
        print("EVALUATING ON LABELED TRAIN SET")
        print("="*60)
        
        total_recall = 0
        results = []
        
        for idx, item in enumerate(train_data, 1):
            query = item['query']
            relevant_urls = set(item['relevant_urls'])
            
            print(f"\n[{idx}/{len(train_data)}] Query: {query[:80]}...")
            print(f"Ground truth has {len(relevant_urls)} relevant assessments")
            
            # Get recommendations from RAG
            recommendations = self.rag.recommend(query, top_k=10)
            recommended_urls = set([rec['url'] for rec in recommendations])
            
            # Calculate recall
            correct_urls = recommended_urls & relevant_urls
            recall = len(correct_urls) / len(relevant_urls) if len(relevant_urls) > 0 else 0
            
            total_recall += recall
            
            print(f"Retrieved {len(correct_urls)}/{len(relevant_urls)} relevant assessments")
            print(f"Recall@10: {recall:.3f}")
            
            if correct_urls:
                print("✓ Correct recommendations:")
                for url in correct_urls:
                    print(f"  - {url}")
            
            missing = relevant_urls - recommended_urls
            if missing:
                print("✗ Missed recommendations:")
                for url in list(missing)[:3]:  # Show first 3
                    print(f"  - {url}")
            
            results.append({
                "query": query,
                "recall": recall,
                "retrieved": len(correct_urls),
                "total_relevant": len(relevant_urls)
            })
        
        # Calculate Mean Recall@10
        mean_recall = total_recall / len(train_data)
        
        print("\n" + "="*60)
        print(f"FINAL RESULTS")
        print("="*60)
        print(f"Mean Recall@10: {mean_recall:.3f}")
        print(f"Average precision: {mean_recall*100:.1f}%")
        print("="*60)
        
        return mean_recall, results
    
    def generate_test_predictions(self, test_csv: str = "data/test_unlabeled.csv", 
                                  output_csv: str = "submission/predictions.csv"):
        """
        Generate predictions for unlabeled test set
        This is what you submit!
        """
        if not os.path.exists(test_csv):
            print(f"Error: {test_csv} not found!")
            return
        
        # Read test queries (just one column: Query)
        df_test = pd.read_csv(test_csv)
        test_queries = df_test['Query'].unique().tolist()
        
        print(f"\nGenerating predictions for {len(test_queries)} test queries...")
        
        all_results = []
        
        for idx, query in enumerate(test_queries, 1):
            print(f"\n[{idx}/{len(test_queries)}] Processing: {query[:80]}...")
            
            # Get recommendations
            recommendations = self.rag.recommend(query, top_k=10)
            
            # Add to results
            for rec in recommendations:
                all_results.append({
                    "Query": query,
                    "Assessment_url": rec['url']
                })
            
            print(f"  → Generated {len(recommendations)} recommendations")
        
        # Save to CSV
        df_results = pd.DataFrame(all_results)
        os.makedirs('submission', exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        
        print(f"\n✓ Predictions saved to {output_csv}")
        print(f"Total rows: {len(df_results)}")
        
        return df_results
    
    def analyze_predictions(self, predictions_csv: str = "submission/predictions.csv"):
        """
        Analyze the generated predictions
        Check for balance, coverage, etc.
        """
        df = pd.read_csv(predictions_csv)
        
        print("\n" + "="*60)
        print("PREDICTION ANALYSIS")
        print("="*60)
        
        # Count queries
        unique_queries = df['Query'].nunique()
        print(f"Number of unique queries: {unique_queries}")
        
        # Recommendations per query
        recs_per_query = df.groupby('Query').size()
        print(f"\nRecommendations per query:")
        print(f"  Min: {recs_per_query.min()}")
        print(f"  Max: {recs_per_query.max()}")
        print(f"  Mean: {recs_per_query.mean():.1f}")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['Query', 'Assessment_url']).sum()
        if duplicates > 0:
            print(f"\n⚠️ Warning: {duplicates} duplicate entries found!")
        else:
            print(f"\n✓ No duplicates found")
        
        print("="*60)

def main():
    evaluator = RAGEvaluator()
    
    # Step 1: Evaluate on labeled train set
    print("\n🔍 STEP 1: EVALUATING ON TRAIN SET")
    mean_recall, results = evaluator.evaluate_on_train()
    
    # Step 2: Generate predictions for test set
    print("\n\n📝 STEP 2: GENERATING TEST PREDICTIONS")
    predictions = evaluator.generate_test_predictions()
    
    # Step 3: Analyze predictions
    print("\n\n📊 STEP 3: ANALYZING PREDICTIONS")
    evaluator.analyze_predictions()
    
    print("\n\n✅ ALL DONE!")
    print(f"Mean Recall@10 on train set: {mean_recall:.3f}")
    print("Test predictions saved to: submission/predictions.csv")

if __name__ == "__main__":
    main()
