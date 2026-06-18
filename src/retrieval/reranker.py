from sentence_transformers import CrossEncoder

# Tải model CrossEncoder (có thể cấu hình thành biến môi trường)
_cross_encoder_model = None

def get_cross_encoder():
    global _cross_encoder_model
    if _cross_encoder_model is None:
        # Sử dụng mô hình hỗ trợ tiếng Việt tốt hoặc đa ngôn ngữ
        _cross_encoder_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    return _cross_encoder_model

def rerank(query: str, candidates: list, k: int = 5):
    """
    Sử dụng CrossEncoder để chấm điểm lại các ứng viên.
    """
    if not candidates:
        return []
        
    encoder = get_cross_encoder()
    
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = encoder.predict(pairs)
    
    # Kết hợp score với document
    scored_docs = list(zip(candidates, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs][:k]

def reranker_search(store, bm25_retriever, query: str, k: int = 5):
    """
    Pipeline: Hybrid Search -> Reranker
    Lấy số lượng ứng viên lớn hơn k (ví dụ k*2) từ hybrid search rồi rerank.
    """
    from src.retrieval.hybrid import hybrid_search
    
    candidates = hybrid_search(store, bm25_retriever, query, k=k*2)
    return rerank(query, candidates, k=k)
