from fastapi import FastAPI
from pydantic import BaseModel
import time
import json
from configs.setting import config
from src.agents.orchestrator import get_orchestrator
from src.indexing.chroma_store import get_vector_store
from src.indexing.bm25_index import load_bm25_index
from src.retrieval.graph import build_graph, graph_search, load_graph
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import reranker_search
from src.tools.retrieval_tools import init_retrieval_components

app = FastAPI(title="AIO Agentic RAG API")

# Khởi tạo các components dùng chung
store = get_vector_store()
bm25 = load_bm25_index()
graph = load_graph()

import os
print("STARTUP DEBUG: LLM_PROVIDER =", os.getenv("LLM_PROVIDER"))
print("STARTUP DEBUG: LLM_MODEL =", os.getenv("LLM_MODEL"))
print("STARTUP DEBUG: OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY")[:15] + "..." if os.getenv("OPENAI_API_KEY") else "None")

init_retrieval_components(store, bm25, graph)
agent = get_orchestrator()

def generate_answer(question: str, docs: list) -> str:
    """
    Sinh câu trả lời từ các tài liệu tìm được bằng LLM.
    """
    if not docs:
        return "Không tìm thấy tài liệu tham khảo phù hợp để trả lời câu hỏi."
        
    from langchain_core.prompts import PromptTemplate
    from src.agents.rag_agent import get_llm
    
    try:
        llm = get_llm()
        
        # Ghép các context
        context_parts = []
        for i, d in enumerate(docs):
            title = d.metadata.get('title', 'Không rõ')
            authority = d.metadata.get('authority', 'Không rõ')
            so_ky_hieu = d.metadata.get('so_ky_hieu', '')
            issue_date = d.metadata.get('issue_date', '')
            source_info = f"Tài liệu {i+1}: {title}"
            if so_ky_hieu:
                source_info += f" (Số ký hiệu: {so_ky_hieu})"
            if authority:
                source_info += f" - Cơ quan ban hành: {authority}"
            if issue_date:
                source_info += f" - Ngày ban hành: {issue_date}"
            
            context_parts.append(f"{source_info}\nNội dung: {d.page_content}")
            
        context = "\n\n".join(context_parts)
        
        prompt_template = """Bạn là trợ lý pháp luật tiếng Việt thông minh và chính xác. 
Nhiệm vụ của bạn là trả lời câu hỏi pháp luật dựa TRÊN NGỮ CẢNH cung cấp dưới đây.
Hãy trả lời chi tiết, chuyên nghiệp và trích dẫn rõ nguồn tài liệu (Điều, Khoản, Tên văn bản, Số ký hiệu, Cơ quan ban hành) nếu có trong ngữ cảnh.
Nếu ngữ cảnh không cung cấp đủ thông tin để trả lời câu hỏi, hãy nói rõ là "Tài liệu tham khảo không cung cấp thông tin cho câu hỏi này". Không tự bịa thông tin không có trong ngữ cảnh.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
        
        prompt = PromptTemplate.from_template(prompt_template)
        chain = prompt | llm
        
        response = chain.invoke({"context": context, "question": question})
        if hasattr(response, "content"):
            return response.content
        return str(response)
    except Exception as e:
        print("DEBUG: error in generate_answer:", e)
        return f"Đã xảy ra lỗi khi tạo câu trả lời: {str(e)}"

class QueryRequest(BaseModel):
    question: str
    strategy: str = "agentic"
    k: int = 5

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    t0 = time.perf_counter()
    q_lower = req.question.lower()
    k = req.k
    
    # Reset/initialize the ContextVar for capturing documents
    from src.tools.retrieval_tools import retrieved_docs_var
    retrieved_docs_var.set([])

    if req.strategy == "naive":
        from src.retrieval.dense import dense_search
        docs = dense_search(store, req.question, k=k)
        answer = generate_answer(req.question, docs)
    elif req.strategy == "hybrid":
        docs = hybrid_search(store, bm25, req.question, k=k, rrf_k=config.retrieval.rrf_k)
        answer = generate_answer(req.question, docs)
    elif req.strategy == "reranker":
        docs = reranker_search(store, bm25, req.question, k=k)
        answer = generate_answer(req.question, docs)
    elif req.strategy == "graph":
        if graph:
            docs = graph_search(store, graph, req.question, k=k, initial_k=3, max_hops=config.retrieval.graph_max_hops)
            answer = generate_answer(req.question, docs)
    elif req.strategy == "agentic":
        # Khởi tạo Agent trực tiếp trong request để tránh xung đột thread/context
        local_agent = get_orchestrator()
        print("DEBUG: Agent Tools:", [t.name for t in local_agent.tools])
        response = local_agent.invoke({"input": req.question})
        answer = response.get("output", "")
        
        # Trích xuất tài liệu từ ContextVar đã được tự động lưu bởi các cuộc gọi tìm kiếm
        retrieved_docs = retrieved_docs_var.get()
                    
        # Fallback nếu agent không gọi công cụ tìm kiếm hoặc không trả về documents nào
        if not retrieved_docs:
            docs = hybrid_search(store, bm25, req.question, k=k, rrf_k=config.retrieval.rrf_k)
        else:
            docs = retrieved_docs

    else:
        # Heuristic routing như trong PDF
        if any(kw in q_lower for kw in ["sửa đổi", "thay thế", "bãi bỏ", "tham chiếu"]) and graph:
            docs = graph_search(store, graph, req.question, k=k)
        elif any(kw in q_lower for kw in ["còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm"]):
            docs = hybrid_search(store, bm25, req.question, k=k)
        else:
            candidates = hybrid_search(store, bm25, req.question, k=k*2)
            docs = reranker_search(store, bm25, req.question, k=k) # Thực ra rerank gọi lại hybrid
        answer = generate_answer(req.question, docs)
            
    latency_ms = (time.perf_counter() - t0) * 1000
    
    # Format docs
    sources = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    
    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms
    }



