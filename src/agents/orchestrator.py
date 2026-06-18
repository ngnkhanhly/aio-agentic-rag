from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agents.rag_agent import get_llm, create_legal_rag_agent
from src.tools.retrieval_tools import (
    dense_search_tool,
    bm25_search_tool,
    hybrid_search_tool,
    rerank_tool,
    graph_traverse_tool,
    generate_answer_tool,
)
from src.agents.ingestion_agent import create_ingestion_agent
from src.tools.ingestion_tools import (
    load_dataset_tool,
    load_relationships_tool,
    clean_docs_tool,
    chunk_docs_tool,
    build_chroma_tool,
    build_bm25_tool,
    build_graph_tool,
)

ORCHESTRATOR_SYSTEM_PROMPT = """Bạn là người điều phối trung tâm (Orchestrator Agent) cho hệ thống trợ lý pháp luật Vietnamese Legal Agentic RAG.

Nhiệm vụ của bạn là nhận yêu cầu từ người dùng và chuyển giao công việc (delegate) cho subagent phù hợp:
1. Nếu người dùng hỏi các câu hỏi về pháp luật, điều khoản, tra cứu luật pháp Việt Nam: Hãy gọi công cụ `legal_rag_subagent_tool` với câu hỏi đầu vào của người dùng. Subagent này sẽ chịu trách nhiệm phân loại, truy xuất và sinh câu trả lời có trích dẫn.
2. Nếu người dùng yêu cầu nạp dữ liệu, tải dataset, làm sạch tài liệu, chia nhỏ chunk hoặc xây dựng lại các chỉ mục (Chroma, BM25, Graph): Hãy gọi công cụ `ingestion_subagent_tool` với hướng dẫn tương ứng của người dùng.

Tuyệt đối không tự mình gọi các công cụ tìm kiếm trực tiếp hoặc tự trả lời các vấn đề chuyên sâu về pháp lý/ingestion mà không sử dụng subagent. Bạn là người quản lý cấp cao điều phối luồng.
Sau khi nhận được câu trả lời từ subagent, hãy trả về kết quả đó cho người dùng mà không cần thay đổi nội dung chuyên môn.
"""

# Lazy-loaded executors to prevent circular imports or premature component loading
_legal_rag_executor = None
_ingestion_executor = None

def get_legal_rag_executor():
    global _legal_rag_executor
    if _legal_rag_executor is None:
        retrieval_tools = [
            dense_search_tool,
            bm25_search_tool,
            hybrid_search_tool,
            rerank_tool,
            graph_traverse_tool,
            generate_answer_tool,
        ]
        _legal_rag_executor = create_legal_rag_agent(retrieval_tools)
    return _legal_rag_executor

def get_ingestion_executor():
    global _ingestion_executor
    if _ingestion_executor is None:
        ingestion_tools = [
            load_dataset_tool,
            load_relationships_tool,
            clean_docs_tool,
            chunk_docs_tool,
            build_chroma_tool,
            build_bm25_tool,
            build_graph_tool,
        ]
        _ingestion_executor = create_ingestion_agent(ingestion_tools)
    return _ingestion_executor

@tool
def legal_rag_subagent_tool(question: str) -> str:
    """
    Gọi Legal RAG Subagent để trả lời các câu hỏi về văn bản pháp lý Việt Nam.
    Sử dụng công cụ này khi người dùng muốn tra cứu luật, hỏi đáp điều khoản luật pháp.
    """
    executor = get_legal_rag_executor()
    response = executor.invoke({"input": question})
    return response.get("output", "Không nhận được phản hồi từ Legal RAG Subagent.")

@tool
def ingestion_subagent_tool(instructions: str) -> str:
    """
    Gọi Ingestion Subagent để thực hiện nạp dữ liệu, tải tài liệu pháp lý từ HuggingFace,
    làm sạch HTML, chia chunk văn bản, và xây dựng các chỉ mục ChromaDB, BM25, Graph.
    Sử dụng công cụ này khi người dùng yêu cầu nạp dữ liệu, index tài liệu hoặc cấu hình hệ thống.
    """
    executor = get_ingestion_executor()
    response = executor.invoke({"input": instructions})
    return response.get("output", "Không nhận được phản hồi từ Ingestion Subagent.")

def get_orchestrator():
    """
    Khởi tạo Orchestrator Agent điều phối chung sử dụng hai subagent tools.
    """
    tools = [legal_rag_subagent_tool, ingestion_subagent_tool]
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )
    return agent_executor
