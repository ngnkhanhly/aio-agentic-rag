import json
from langchain_core.tools import tool
from configs.setting import config
from src.retrieval.dense import dense_search
from src.retrieval.bm25 import bm25_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.graph import graph_search
from src.agents.rag_agent import get_llm
import contextvars

# ContextVar to capture all retrieved documents during a request
retrieved_docs_var = contextvars.ContextVar("retrieved_docs", default=[])

def _capture_docs(docs):
    try:
        current = retrieved_docs_var.get()
        new_list = list(current)
        for doc in docs:
            doc_id = doc.metadata.get("doc_id", "")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            if not any(
                d.metadata.get("doc_id", "") == doc_id and d.metadata.get("chunk_index", 0) == chunk_idx
                for d in new_list
            ):
                new_list.append(doc)
        retrieved_docs_var.set(new_list)
    except Exception as e:
        print("DEBUG: Error capturing docs in ContextVar:", e)

_store = None
_bm25 = None
_graph = None

def init_retrieval_components(store, bm25, graph):
    global _store, _bm25, _graph
    _store = store
    _bm25 = bm25
    _graph = graph

def _doc_to_dict(doc) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata
    }

@tool
def dense_search_tool(query: str, k: int = None, metadata_filter_json: str = "") -> str:
    """
    Perform dense vector search in Chroma vector database.
    Use this for general semantic queries where the keywords might not exactly match the text.
    Args:
        query: Search query string.
        k: Number of results to retrieve.
        metadata_filter_json: Optional JSON string of metadata filter dict (e.g. {"doc_type": "Chỉ thị"}).
    Returns:
        JSON string list of retrieved documents.
    """
    if not _store:
        return "Vector store is not initialized."
    val_k = k or getattr(config.retrieval, "k", 5)
    flt = json.loads(metadata_filter_json) if metadata_filter_json else None
    docs = dense_search(_store, query, k=val_k, metadata_filter=flt)
    _capture_docs(docs)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)

@tool
def bm25_search_tool(query: str, k: int = None) -> str:
    """
    Perform BM25 keyword search.
    Use this when queries contain specific names, numbers, or legal codes.
    Args:
        query: Search query string.
        k: Number of results to retrieve.
    Returns:
        JSON string list of retrieved documents.
    """
    if not _bm25:
        return "BM25 index is not initialized."
    val_k = k or getattr(config.retrieval, "k", 5)
    docs = bm25_search(_bm25, query, k=val_k)
    _capture_docs(docs)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)

@tool
def hybrid_search_tool(query: str, k: int = None, metadata_filter_json: str = "") -> str:
    """
    Perform hybrid BM25 + dense retrieval with RRF fusion.
    Use this for queries that contain both semantic meaning and specific legal keywords.
    Args:
        query: Search query string.
        k: Number of results to retrieve.
        metadata_filter_json: Optional JSON string of metadata filter dict.
    Returns:
        JSON string list of retrieved documents.
    """
    if not _store or not _bm25:
        return "Retrieval components not fully initialized."
    val_k = k or getattr(config.retrieval, "k", 5)
    # Note: RRF fusion doesn't support filter directly unless implemented, but we match reference signature
    docs = hybrid_search(_store, _bm25, query, k=val_k, rrf_k=getattr(config.retrieval, "rrf_k", 60))
    _capture_docs(docs)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)

@tool
def rerank_tool(query: str, docs_json: str = "", k: int = None) -> str:
    """
    Re-rank a list of retrieved documents with a CrossEncoder model.
    Use this to sort the candidates more accurately for high-precision queries.
    Args:
        query: Original user question.
        docs_json: Optional JSON string list of retrieved documents to re-rank. If not provided or if JSON parsing fails, it will use the captured documents.
        k: Top-k to return after reranking.
    Returns:
        JSON string list of reranked documents.
    """
    from langchain_core.documents import Document
    docs_list = []
    if docs_json:
        try:
            docs_list = json.loads(docs_json)
        except Exception as e:
            print("DEBUG: Warning: JSON parsing failed in rerank_tool, falling back to captured docs:", e)
            
    if not docs_list:
        captured = retrieved_docs_var.get()
        docs = list(captured)
    else:
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs_list]
        
    val_k = k or getattr(config.retrieval, "k", 5)
    reranked = rerank(query, docs, k=val_k)
    retrieved_docs_var.set(reranked)
    return json.dumps([_doc_to_dict(d) for d in reranked], ensure_ascii=False)

@tool
def graph_traverse_tool(query: str, k: int = None, initial_k: int = 3, max_hops: int = None) -> str:
    """
    Retrieve documents using graph-guided multi-hop retrieval.
    Seed with dense search, then expand via relationship graph edges.
    Use this for multi-hop questions about document history, amendments, replacements or legal hierarchy.
    """
    if not _store or not _graph:
        return "Retrieval components not fully initialized."
    val_k = k or getattr(config.retrieval, "k", 5)
    val_hops = max_hops or getattr(config.retrieval, "graph_max_hops", 2)
    docs = graph_search(
        _store, _graph, query,
        k=val_k,
        initial_k=initial_k,
        max_hops=val_hops
    )
    _capture_docs(docs)
    return json.dumps([_doc_to_dict(d) for d in docs], ensure_ascii=False)

@tool
def generate_answer_tool(query: str, docs_json: str = "") -> str:
    """
    Generate an answer from retrieved documents using the LLM.
    Use this to formulate the final grounded answer with citations.
    Args:
        query: User question.
        docs_json: Optional JSON string list of retrieved documents. If not provided or if JSON parsing fails, it will use the captured documents.
    Returns:
        Generated answer string with citations.
    """
    docs_list = []
    if docs_json:
        try:
            docs_list = json.loads(docs_json)
        except Exception as e:
            print("DEBUG: Warning: JSON parsing of docs_json failed in generate_answer_tool, falling back to captured docs:", e)
            
    if not docs_list:
        captured = retrieved_docs_var.get()
        docs_list = [{"metadata": d.metadata, "page_content": d.page_content} for d in captured]
    else:
        from langchain_core.documents import Document
        final_docs = [
            Document(page_content=d.get("page_content", ""), metadata=d.get("metadata", {}))
            for d in docs_list
        ]
        retrieved_docs_var.set(final_docs)
        
    context_parts = []
    for i, doc in enumerate(docs_list, 1):
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        title = metadata.get("title", "Không rõ")
        doc_id = metadata.get("doc_id", "unknown")
        so_ky_hieu = metadata.get("so_ky_hieu", "")
        authority = metadata.get("authority", "")
        
        citation = f"Tài liệu [{i}]: {title}"
        if so_ky_hieu:
            citation += f" (Số ký hiệu: {so_ky_hieu})"
        if authority:
            citation += f" - Ban hành bởi: {authority}"
            
        page_content = doc.get("page_content", "") if isinstance(doc, dict) else ""
        context_parts.append(f"{citation}\nNội dung: {page_content}")
        
    context = "\n\n".join(context_parts)
    llm = get_llm()
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam.\n"
                "Hãy trả lời câu hỏi chi tiết, chuyên nghiệp và trích dẫn rõ nguồn tài liệu (Điều, Khoản, Tên văn bản, Số ký hiệu) có trong ngữ cảnh.\n"
                "Nếu không tìm thấy thông tin trong tài liệu tham khảo, hãy nói rõ."
            ),
        },
        {
            "role": "user",
            "content": f"Tài liệu tham khảo:\n\n{context}\n\nCâu hỏi: {query}",
        },
    ]
    response = llm.invoke(messages)
    return response.content
