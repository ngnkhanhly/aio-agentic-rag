def rrf(dense_docs, bm25_docs, k: int = 5, rrf_k: int = 60):
    """
    Reciprocal Rank Fusion (RRF) để kết hợp kết quả từ 2 phương pháp.
    """
    scores = {}
    docs_dict = {}
    
    for rank, doc in enumerate(dense_docs):
        # Tạo khóa duy nhất cho document dựa trên doc_id và chunk_index
        doc_id = doc.metadata.get("doc_id", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        key = f"{doc_id}_{chunk_idx}"
        
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        docs_dict[key] = doc
        
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc.metadata.get("doc_id", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        key = f"{doc_id}_{chunk_idx}"
        
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        docs_dict[key] = doc
        
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [docs_dict[key] for key, _ in sorted_scores][:k]

def hybrid_search(store, bm25_retriever, query: str, k: int = 5, rrf_k: int = 60):
    """
    Kết hợp dense_search và BM25 search.
    """
    from src.retrieval.dense import dense_search
    
    dense_docs = dense_search(store, query, k=k*2)
    bm25_docs = bm25_retriever.invoke(query)
    
    return rrf(dense_docs, bm25_docs, k=k, rrf_k=rrf_k)
