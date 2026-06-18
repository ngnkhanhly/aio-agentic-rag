import gradio as gr
import requests

API_URL = "http://localhost:8001/query"
STRATEGIES = ["naive", "hybrid", "reranker", "graph", "agentic"]

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Apply modern font and elegant clean light slate background */
body, .gradio-container {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: radial-gradient(circle at top, #f8fafc 0%, #f1f5f9 100%) !important;
    color: #1e293b !important;
}

/* Header container styling */
.header-box {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}

.header-title {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800;
    font-size: 2.8rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}

.header-subtitle {
    font-size: 1.1rem;
    color: #475569;
    max-width: 700px;
    margin: 0 auto;
    font-weight: 400;
    line-height: 1.6;
}

/* Section labels */
.section-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #4f46e5;
    padding-left: 10px;
}

/* Custom latency badge style */
.latency-wrapper {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

.latency-badge {
    background: rgba(79, 70, 229, 0.08);
    border: 1px solid rgba(79, 70, 229, 0.15);
    color: #4f46e5;
    border-radius: 9999px;
    padding: 6px 16px;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600;
    font-size: 0.9rem;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.05);
}

/* Table overrides for a modern grid look */
.gr-dataframe table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    background-color: #ffffff !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
}

.gr-dataframe th {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    border-bottom: 1px solid #cbd5e1 !important;
    padding: 12px 16px !important;
}

.gr-dataframe td {
    padding: 12px 16px !important;
    border-bottom: 1px solid #e2e8f0 !important;
    color: #334155 !important;
}

.gr-dataframe tr:hover td {
    background-color: #f8fafc !important;
}

/* Markdown response output panel */
.markdown-box {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: #1e293b !important;
    line-height: 1.7 !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05) !important;
}

/* Elegant primary buttons styling */
button.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-family: 'Outfit', sans-serif !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.3) !important;
}

/* Clean UI components card style */
.block {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05) !important;
}
"""

def do_query(question, strategy, k):
    try:
        response = requests.post(API_URL, json={
            "question": question,
            "strategy": strategy,
            "k": k
        })
        response.raise_for_status()
        data = response.json()
        
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        latency = data.get("latency_ms", 0)
        
        sources_list = [
            [
                s.get("metadata", {}).get("doc_id", "Unknown"),
                s.get("metadata", {}).get("title", "Unknown"),
                s.get("metadata", {}).get("authority", "Unknown"),
                s.get("metadata", {}).get("issue_date", "Unknown"),
                s.get("page_content", "")
            ]
            for s in sources
        ]
        
        latency_html = f"""
        <div class="latency-wrapper">
            <div class="latency-badge">
                <span>⏱️</span>
                <span>Thời gian xử lý: {latency:.2f} ms</span>
            </div>
        </div>
        """
        
        return answer, sources_list, latency_html
    except Exception as e:
        error_html = f"""
        <div class="latency-wrapper">
            <div class="latency-badge" style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); color: #dc2626;">
                <span>❌</span>
                <span>Lỗi: {str(e)}</span>
            </div>
        </div>
        """
        return f"Đã xảy ra lỗi khi truy vấn: {str(e)}", [], error_html

custom_head = """
<script>
function removeDark() {
    document.documentElement.classList.remove('dark');
    document.body.classList.remove('dark');
}
removeDark();
// Monitor for changes
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
        }
        if (document.body.classList.contains('dark')) {
            document.body.classList.remove('dark');
        }
    });
});
observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
</script>
"""

with gr.Blocks(title="AIO Agentic RAG", css=custom_css, head=custom_head) as demo:
    gr.HTML("""
    <div class="header-box">
        <h1 class="header-title">Vietnamese Legal Agentic RAG</h1>
        <p class="header-subtitle">
            Hệ thống trợ lý pháp luật thông minh sử dụng kiến trúc tác tử 2 tầng kết hợp Vector Index, BM25 Keyword Index và Đồ thị quan hệ (Graph Index).
        </p>
    </div>
    """)
    
    with gr.Tab("Tra Cứu Pháp Luật (Query)"):
        with gr.Row():
            with gr.Column(scale=4):
                question_in = gr.Textbox(
                    label="Câu hỏi pháp luật", 
                    placeholder="Nhập câu hỏi của bạn tại đây... (Ví dụ: Chỉ thị 26/1999/CT-UB quy định về vấn đề gì?)",
                    lines=3
                )
            with gr.Column(scale=1):
                strategy_in = gr.Dropdown(
                    choices=STRATEGIES, 
                    value="agentic", 
                    label="Chiến lược truy xuất"
                )
                k_in = gr.Slider(
                    minimum=1, 
                    maximum=20, 
                    value=5, 
                    step=1, 
                    label="Số lượng tài liệu (k)"
                )
                
        ask_btn = gr.Button("Gửi Câu Hỏi", variant="primary")
        
        info_html = gr.HTML("""
        <div class="latency-wrapper">
            <div class="latency-badge">
                <span>⏱️</span>
                <span>Đang chờ câu hỏi...</span>
            </div>
        </div>
        """)
        
        gr.HTML('<div class="section-title">Câu Trả Lời</div>')
        answer_box = gr.Markdown(elem_classes="markdown-box")
        
        gr.HTML('<div class="section-title">Nguồn Trích Dẫn (Sources)</div>')
        sources_table = gr.Dataframe(
            headers=["doc_id", "title", "authority", "issued", "excerpt"], 
            label=""
        )
        
        ask_btn.click(
            fn=do_query,
            inputs=[question_in, strategy_in, k_in],
            outputs=[answer_box, sources_table, info_html]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
