import networkx as nx
import pickle
import os

GRAPH_PATH = "graph.pkl"

def save_graph(G: nx.DiGraph, path: str = GRAPH_PATH):
    with open(path, "wb") as f:
        pickle.dump(G, f)

def load_graph(path: str = GRAPH_PATH) -> nx.DiGraph:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def build_graph(relationships: list[dict]) -> nx.DiGraph:
    """
    Xây dựng đồ thị có hướng biểu diễn quan hệ giữa các văn bản pháp luật.
    relationships: danh sách các dictionary, ví dụ chứa 'doc_id', 'other_doc_id', 'relationship'.
    """
    G = nx.DiGraph()
    for row in relationships:
        src = str(row.get("doc_id", ""))
        dst = str(row.get("other_doc_id", ""))
        rel = row.get("relationship", "")
        if src and dst:
            G.add_edge(src, dst, rel_type=rel)
    return G

def graph_search(store, graph: nx.DiGraph, query: str, k: int = 5, initial_k: int = 3, max_hops: int = 2):
    """
    Tìm kiếm seed documents bằng dense_search, sau đó mở rộng theo graph edges (cả in và out).
    """
    # Import ở đây để tránh circular dependency
    from src.retrieval.dense import dense_search
    
    seed_docs = dense_search(store, query, k=initial_k)
    seed_ids = {d.metadata.get("doc_id", "") for d in seed_docs}
    
    reachable, frontier = set(seed_ids), set(seed_ids)
    for _ in range(max_hops):
        nxt: set[str] = set()
        for node in frontier:
            if node not in graph:
                continue
            
            # Thu thập các node đến (out_edges)
            nxt |= {nb for _, nb, _ in graph.out_edges(node, data=True) if nb not in reachable}
            # Thu thập các node đi (in_edges)
            nxt |= {nb for nb, _, _ in graph.in_edges(node, data=True) if nb not in reachable}
            
        reachable |= nxt
        frontier = nxt
        if not frontier:
            break
            
    extra_ids = reachable - seed_ids
    extra_docs = []
    if extra_ids:
        # Trong Chroma, truy vấn metadata có chứa list extra_ids
        # Dùng db get hoặc tìm kiếm qua filter
        # Để đơn giản, ta dùng similarity_search kèm filter in
        extra_docs = store.similarity_search(
            query, 
            k=k * 2, 
            filter={"doc_id": {"$in": list(extra_ids)}}
        )
        
    # Loại bỏ trùng lặp và giữ nguyên thứ tự seed
    seen = set()
    deduped = []
    for doc in list(seed_docs) + extra_docs:
        # Key deduplication dựa trên doc_id và chunk_index
        doc_id = doc.metadata.get("doc_id", "")
        chunk_idx = doc.metadata.get("chunk_index", 0)
        key = f"{doc_id}_{chunk_idx}"
        
        if key not in seen:
            seen.add(key)
            deduped.append(doc)
            
    return deduped[:k]
