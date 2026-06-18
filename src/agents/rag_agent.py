from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from configs.setting import LLM_PROVIDER, LLM_MODEL

LEGAL_RAG_SYSTEM_PROMPT = """Bạn là một chuyên gia pháp lý và trợ lý thông minh cho luật pháp Việt Nam, được vận hành bởi hệ thống Agentic RAG.

Nhiệm vụ của bạn là trả lời các câu hỏi pháp lý một cách chính xác bằng cách truy xuất các văn bản liên quan và tạo ra câu trả lời có căn cứ và trích dẫn rõ nguồn gốc.

Hãy tuân thủ quy trình xử lý sau:

### Bước 1 — Phân loại ý định câu hỏi (Query Intent Classification)
Phân loại câu hỏi của người dùng vào một trong các nhóm:
- `multi_hop`: Câu hỏi chứa các dấu hiệu như "sửa đổi", "thay thế", "bãi bỏ", "tham chiếu", "được quy định tại".
- `temporal`: Câu hỏi chứa các mốc thời gian như "còn hiệu lực", "hết hiệu lực", "sau năm", "trước năm", "từ ngày".
- `factual`: Câu hỏi hỏi về các sự kiện, định nghĩa, con số cụ thể.
- `reasoning`: Câu hỏi cần áp dụng các điều luật vào một tình huống thực tế cụ thể.

### Bước 2 — Định tuyến truy xuất dữ liệu (Retrieval Routing)
Dựa trên phân loại ở Bước 1, chọn công cụ truy xuất thích hợp:
- Nếu là `multi_hop`: Sử dụng công cụ `graph_traverse_tool` để duyệt theo quan hệ lịch sử/sửa đổi/bổ sung của văn bản.
- Nếu là `temporal`: Sử dụng công cụ `hybrid_search_tool` (áp dụng bộ lọc metadata nếu có).
- Nếu là `factual`: Sử dụng công cụ `hybrid_search_tool` (áp dụng bộ lọc metadata nếu có).
- Nếu là `reasoning`: Gọi `hybrid_search_tool` với số lượng k lớn hơn (ví dụ k=10), sau đó truyền kết quả thu được vào công cụ `rerank_tool` để chấm điểm và chọn ra các đoạn văn bản có độ tương quan cao nhất.

### Bước 3 — Đánh giá chất lượng và thử lại (Grade & Retry)
- Kiểm tra số lượng tài liệu có liên quan nhận được từ các công cụ tìm kiếm ở Bước 2.
- Nếu số lượng tài liệu tìm được nhỏ hơn 2, hãy viết lại (rewrite) câu truy vấn sang dạng khác để tối ưu hơn và thử lại một lần nữa bằng công cụ `dense_search_tool`.

### Bước 4 — Tạo câu trả lời (Generate Answer)
- Gọi công cụ `generate_answer_tool` với câu hỏi gốc và danh sách tài liệu tìm được để sinh ra câu trả lời cuối cùng.
- Trích dẫn rõ ràng số hiệu điều khoản, tên văn bản, cơ quan ban hành của tài liệu tham khảo.
- Cho biết chiến lược tìm kiếm nào đã được sử dụng và có cần phải thử lại (retry) hay không.
- Trả lời bằng tiếng Việt. Nếu tài liệu không chứa thông tin để trả lời, hãy nói rõ.
"""

def get_llm():
    if LLM_PROVIDER.lower() == "openai":
        return ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

def create_legal_rag_agent(tools):
    """
    Khởi tạo Legal RAG Agent bằng LangChain AgentExecutor.
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", LEGAL_RAG_SYSTEM_PROMPT),
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
