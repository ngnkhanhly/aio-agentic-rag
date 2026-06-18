def bm25_search(bm25_retriever, query: str, k: int = 5):
    """
    Tìm kiếm văn bản theo từ khóa bằng chỉ mục BM25.
    """
    if bm25_retriever is None:
        return []
    bm25_retriever.k = k
    return bm25_retriever.invoke(query)
