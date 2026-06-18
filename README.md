# AIO Agentic RAG - Hệ Thống Trợ Lý Pháp Luật Lâm Đồng

Hệ thống hỏi đáp pháp luật thông minh (QA) sử dụng kiến trúc **Tác tử 2 tầng (Two-Tier Agents)**, kết hợp tìm kiếm kết hợp (**Dense & Sparse Hybrid Search**) và Đồ thị quan hệ pháp lý (**Relational Graph Index**). Giao diện hiển thị được thiết kế theo phong cách **Premium Light Theme** trực quan và hiện đại.

---

## 🏗️ Kiến trúc Hệ thống

Hệ thống được thiết kế theo các thành phần cốt lõi:

### 1. Kiến trúc Tác tử 2 tầng (Two-Tier Agents)
*   **Orchestrator Agent**: Đóng vai trò điều phối trung tâm. Nhận câu hỏi từ người dùng, phân tích mục tiêu và định tuyến thông minh đến một trong hai tác tử con (Subagents):
    *   **Ingestion Agent**: Tác tử nạp dữ liệu. Tự động lập kế hoạch và thực thi chuỗi công cụ tải dữ liệu, tiền xử lý, băm nhỏ (chunking), lập chỉ mục dense vector, BM25 và xây dựng đồ thị quan hệ.
    *   **RAG QA Agent**: Tác tử hỏi đáp. Thực hiện phân loại ý định của câu hỏi (Factual, Reasoning, Temporal, Multi-hop), lựa chọn công cụ truy xuất tối ưu, tổng hợp câu trả lời và tự động đánh giá chất lượng câu trả lời trước khi phản hồi (Self-Grading & Loop Retry).

### 2. Các chỉ mục tìm kiếm (Multi-Indexing)
*   **Dense Vector Index (ChromaDB)**: Truy xuất các đoạn văn bản có sự tương đồng ngữ nghĩa cao.
*   **Sparse Keyword Index (BM25)**: Đảm bảo độ chính xác khi tìm kiếm các từ khóa đặc biệt như số hiệu văn bản (ví dụ: *26/1999/CT-UB*), ngày ban hành hoặc cơ quan thẩm quyền.
*   **Relational Graph Index (NetworkX)**: Kết nối các văn bản pháp luật dựa trên:
    *   Cơ quan ban hành (Authority).
    *   Thời gian ban hành (Issued date).
    *   Mối liên hệ tham chiếu chéo giữa các điều khoản.

---

## 📂 Cấu trúc Thư mục

```text
├── api/
│   └── main.py              # FastAPI Server (Cung cấp endpoint /query)
├── configs/
│   └── setting.py           # Quản lý cấu hình & biến môi trường
├── scripts/
│   └── ingest.py            # Chạy pipeline nạp dữ liệu sử dụng Ingestion Agent
├── src/
│   ├── agents/
│   │   ├── orchestrator.py  # Orchestrator Agent điều phối chính
│   │   ├── rag_agent.py     # RAG QA Agent với khả năng tự chấm điểm câu trả lời
│   │   └── ingestion_agent.py# Ingestion Agent lập kế hoạch xây dựng chỉ mục
│   ├── indexing/
│   │   ├── chroma_store.py  # Quản lý cơ sở dữ liệu vector ChromaDB
│   │   ├── bm25_index.py    # Xây dựng chỉ mục BM25
│   │   └── embeddings.py    # Khởi tạo mô hình nhúng (Embeddings)
│   ├── ingestion/
│   │   ├── loader.py        # Đọc dữ liệu từ HuggingFace Hub hoặc tệp nội bộ
│   │   ├── cleaner.py       # Làm sạch văn bản, chuẩn hóa tiếng Việt, loại bỏ HTML
│   │   └── chunker.py       # Phân đoạn văn bản đè gối thông minh
│   ├── retrieval/
│   │   ├── dense.py         # Tìm kiếm Dense Vector
│   │   ├── bm25.py          # Tìm kiếm BM25
│   │   ├── hybrid.py        # Tìm kiếm kết hợp (Dense + BM25)
│   │   ├── reranker.py      # Tái sắp xếp tài liệu sử dụng Cross-Encoder
│   │   └── graph.py         # Tìm kiếm nâng rộng trên Đồ thị quan hệ
│   └── tools/
│       ├── ingestion_tools.py # Danh sách các công cụ dành cho Ingestion Agent
│       └── retrieval_tools.py # Danh sách các công cụ dành cho RAG Agent (thu giữ context)
├── ui/
│   └── app.py               # Giao diện Gradio Web UI (Phong cách Premium Light Theme)
├── .env.example             # Bản mẫu cấu hình biến môi trường
├── .gitignore               # Tệp cấu hình các mục Git bỏ qua
└── requirements.txt         # Danh sách thư viện Python cần thiết
```

---

## 🛠️ Cài đặt & Khởi chạy

### 1. Chuẩn bị Môi trường
Tải mã nguồn và tạo môi trường ảo Python 3.10:
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo (Windows)
.venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Cấu hình biến môi trường
Sao chép tệp `.env.example` thành `.env` và điền đầy đủ các thông tin:
```bash
cp .env.example .env
```
Nội dung tệp `.env`:
```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3. Nạp Dữ liệu & Xây dựng Chỉ mục
Sử dụng Tác tử nạp dữ liệu để tự động hóa toàn bộ quá trình nạp và lập chỉ mục:
```bash
python scripts/ingest.py
```
*Tệp chỉ mục BM25 và Đồ thị quan hệ sẽ được lưu dưới dạng `.pkl`, cơ sở dữ liệu vector sẽ được lưu tại thư mục `chroma_db/`.*

### 4. Khởi chạy FastAPI Server
Khởi chạy dịch vụ hỏi đáp:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```
*Tài liệu Swagger UI sẽ khả dụng tại địa chỉ `http://localhost:8001/docs`.*

### 5. Khởi chạy Gradio Web UI
Giao diện người dùng tra cứu với giao diện **Light Theme** hiện đại:
```bash
python ui/app.py
```
*Giao diện Web UI sẽ khả dụng tại địa chỉ `http://localhost:7860/`.*
