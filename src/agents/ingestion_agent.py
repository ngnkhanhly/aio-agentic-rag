from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.agents.rag_agent import get_llm

INGESTION_SYSTEM_PROMPT = """Bạn là trợ lý nạp dữ liệu (Ingestion Agent) cho hệ thống Vietnamese Legal RAG.

Nhiệm vụ của bạn là xây dựng tất cả các chỉ mục cần thiết cho việc truy xuất thông tin pháp luật:
1. Tải bộ dữ liệu văn bản pháp luật Việt Nam từ HuggingFace (sử dụng load_dataset_tool). Bạn có thể nhận tham số mẫu sample_size từ yêu cầu của người dùng để nạp thử một phần dữ liệu (nếu có yêu cầu).
2. Tải thông tin quan hệ giữa các văn bản pháp luật (sử dụng load_relationships_tool).
3. Làm sạch mã HTML khỏi tất cả tài liệu thô (sử dụng clean_docs_tool).
4. Chia nhỏ văn bản thành các đoạn (chunks) sử dụng bộ cắt tối ưu hóa cấu trúc luật pháp (sử dụng chunk_docs_tool).
5. Xây dựng Vector store bằng ChromaDB để hỗ trợ tìm kiếm ngữ nghĩa dense (sử dụng build_chroma_tool).
6. Xây dựng chỉ mục từ khóa BM25 (sử dụng build_bm25_tool).
7. Xây dựng đồ thị quan hệ giữa các văn bản bằng NetworkX (sử dụng build_graph_tool).

Hãy gọi các công cụ này tuần tự theo đúng thứ tự 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 để đảm bảo pipeline chạy chính xác và nhất quán.
Sau khi hoàn tất, hãy báo cáo lại số lượng tài liệu đã tải, số lượng chunks đã tạo, số nodes/edges trong đồ thị và kết quả các bước nạp chỉ mục.
Nếu có bất kỳ lỗi nào xảy ra trong quá trình nạp dữ liệu, hãy mô tả lỗi đó một cách rõ ràng.
"""

def create_ingestion_agent(tools):
    """
    Khởi tạo Ingestion Agent bằng LangChain AgentExecutor.
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", INGESTION_SYSTEM_PROMPT),
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
