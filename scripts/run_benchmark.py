import sys
import os
import time
import json
import argparse
import pandas as pd
from datetime import datetime

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from configs.setting import config
from src.indexing.chroma_store import get_vector_store
from src.indexing.bm25_index import load_bm25_index
from src.retrieval.graph import load_graph, graph_search
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import reranker_search
from src.evaluation.metrics import recall_at_k, ndcg_at_k
from src.tools.retrieval_tools import init_retrieval_components, retrieved_docs_var
from src.agents.orchestrator import get_orchestrator

def run_evaluation(strategies, sample_size=None):
    # Load evaluation set
    eval_set_path = "evaluation_set.json"
    if not os.path.exists(eval_set_path):
        print("evaluation_set.json not found! Run scripts/build_eval_set.py first.")
        sys.exit(1)
        
    with open(eval_set_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
        
    if sample_size:
        questions = questions[:sample_size]
        
    print(f"Running evaluation on {len(questions)} queries...")
    
    # Initialize components
    store = get_vector_store()
    bm25 = load_bm25_index()
    graph = load_graph()
    init_retrieval_components(store, bm25, graph)
    
    results = []
    
    for strategy in strategies:
        print(f"\nEvaluating strategy: {strategy}...")
        total_recall_5 = 0.0
        total_ndcg_10 = 0.0
        total_latency_ms = 0.0
        
        for q_item in questions:
            question = q_item["question"]
            relevant_ids = q_item["expected_doc_ids"]
            
            # Reset ContextVar
            retrieved_docs_var.set([])
            
            t0 = time.perf_counter()
            retrieved_docs = []
            
            try:
                if strategy == "naive":
                    retrieved_docs = dense_search(store, question, k=5)
                elif strategy == "hybrid":
                    retrieved_docs = hybrid_search(store, bm25, question, k=5, rrf_k=config.retrieval.rrf_k)
                elif strategy == "reranker":
                    retrieved_docs = reranker_search(store, bm25, question, k=5)
                elif strategy == "graph":
                    if graph:
                        retrieved_docs = graph_search(store, graph, question, k=5, initial_k=3, max_hops=config.retrieval.graph_max_hops)
                elif strategy == "agentic":
                    agent = get_orchestrator()
                    response = agent.invoke({"input": question})
                    retrieved_docs = retrieved_docs_var.get()
                    if not retrieved_docs:
                        # fallback
                        retrieved_docs = hybrid_search(store, bm25, question, k=5, rrf_k=config.retrieval.rrf_k)
            except Exception as e:
                print(f"Error evaluating query '{question}' with strategy '{strategy}': {e}")
                
            latency_ms = (time.perf_counter() - t0) * 1000
            
            retrieved_ids = [d.metadata.get("doc_id", "") for d in retrieved_docs]
            
            # Compute metrics
            r5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
            n10 = ndcg_at_k(retrieved_ids, relevant_ids, k=10)
            
            total_recall_5 += r5
            total_ndcg_10 += n10
            total_latency_ms += latency_ms
            
        avg_r5 = total_recall_5 / len(questions)
        avg_n10 = total_ndcg_10 / len(questions)
        avg_latency = total_latency_ms / len(questions)
        
        print(f"  Recall@5: {avg_r5:.4f}")
        print(f"  NDCG@10: {avg_n10:.4f}")
        print(f"  Latency: {avg_latency:.2f} ms")
        
        results.append({
            "Strategy": strategy,
            "Recall@5": round(avg_r5, 4),
            "NDCG@10": round(avg_n10, 4),
            "Latency (ms)": round(avg_latency, 2)
        })
        
    df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("evaluation", exist_ok=True)
    csv_path = f"evaluation/benchmark_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark results to {csv_path}")
    print("\nSummary Table:")
    print(df.to_string(index=False))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="all", choices=["naive", "hybrid", "reranker", "graph", "agentic", "all"])
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    
    if args.strategy == "all":
        strategies = ["naive", "hybrid", "reranker", "graph", "agentic"]
    else:
        strategies = [args.strategy]
        
    run_evaluation(strategies, sample_size=args.sample)

if __name__ == "__main__":
    main()
