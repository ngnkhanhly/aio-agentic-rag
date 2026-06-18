import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

EVAL_SET = [
    {
        "question": "Văn bản nào nói về việc tăng cường quản lý đất lâm nghiệp?",
        "expected_doc_ids": ["4262"]
    },
    {
        "question": "UBND tỉnh Lâm Đồng chỉ thị về việc tăng cường quản lý đất lâm nghiệp thế nào?",
        "expected_doc_ids": ["4262"]
    },
    {
        "question": "Tìm văn bản quy định về tăng cường quản lý đất lâm nghiệp ở Lâm Đồng.",
        "expected_doc_ids": ["4262"]
    },
    {
        "question": "Đơn vị nào chịu trách nhiệm quản lý đất lâm nghiệp theo Chỉ thị 26/1999/CT-UB?",
        "expected_doc_ids": ["4262"]
    },
    {
        "question": "Văn bản nào nói về việc tăng cường công tác quản lý nhà, đất của các tôn giáo ở Lâm Đồng?",
        "expected_doc_ids": ["4266"]
    },
    {
        "question": "Chỉ thị 23/1999/CT-UB điều chỉnh những đối tượng tôn giáo nào?",
        "expected_doc_ids": ["4266"]
    },
    {
        "question": "Quy định về việc quản lý nhà đất tôn giáo tại Lâm Đồng nằm ở văn bản nào?",
        "expected_doc_ids": ["4266"]
    },
    {
        "question": "Văn bản nào nói về việc chuyển giao nhiệm vụ quản lý nhà nước về đất lâm nghiệp?",
        "expected_doc_ids": ["4260"]
    },
    {
        "question": "Chuyển giao quản lý nhà nước đất lâm nghiệp từ Sở Nông nghiệp sang Sở Địa chính được quy định ở đâu?",
        "expected_doc_ids": ["4260"]
    },
    {
        "question": "Quyết định chuyển giao đất lâm nghiệp sang Sở Địa chính Lâm Đồng ban hành khi nào?",
        "expected_doc_ids": ["4260"]
    },
    {
        "question": "Văn bản nào quy định về việc quản lý kinh doanh mặt hàng rượu?",
        "expected_doc_ids": ["4264"]
    },
    {
        "question": "Lâm Đồng quản lý kinh doanh mặt hàng rượu theo văn bản pháp luật nào?",
        "expected_doc_ids": ["4264"]
    },
    {
        "question": "Quy định về kinh doanh rượu tại tỉnh Lâm Đồng.",
        "expected_doc_ids": ["4264"]
    },
    {
        "question": "Văn bản nào quản lý thống nhất việc thu phí vào cổng tham quan danh lam thắng cảnh ở Lâm Đồng?",
        "expected_doc_ids": ["4281"]
    },
    {
        "question": "Quy định thu phí tham quan danh lam thắng cảnh tại tỉnh Lâm Đồng ban hành bởi cơ quan nào?",
        "expected_doc_ids": ["4281"]
    },
    {
        "question": "Tìm văn bản về mức thu phí vào cổng tham quan các khu du lịch ở Lâm Đồng.",
        "expected_doc_ids": ["4281"]
    }
]

def main():
    out_path = Path("evaluation_set.json")
    out_path.write_text(json.dumps(EVAL_SET, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã tạo file bộ câu hỏi đánh giá gold set tại: {out_path} ({len(EVAL_SET)} câu hỏi)")

if __name__ == "__main__":
    main()
