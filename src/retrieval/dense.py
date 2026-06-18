def dense_search(store, query: str, k: int = 5, metadata_filter: dict = None):
    """
    Tìm kiếm dựa trên độ tương đồng vector (dense retrieval) bằng ChromaDB.
    """
    if metadata_filter:
        docs = store.similarity_search(query, k=k, filter=metadata_filter)
    else:
        docs = store.similarity_search(query, k=k)
    return docs
