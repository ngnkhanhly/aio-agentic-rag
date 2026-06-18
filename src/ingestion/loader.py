from langchain_core.documents import Document

def load_documents(dataset_rows, meta_lookup, sample_size=None):
    """
    Tải dữ liệu từ HuggingFace dataset và chuyển thành danh sách Document.
    """
    docs = []
    
    if sample_size:
        rows = dataset_rows[:sample_size]
    else:
        rows = dataset_rows
        
    for row in rows:
        doc_id = str(row.get("id", ""))
        meta = meta_lookup.get(doc_id, {})
        
        # Ánh xạ trường metadata tiếng Việt sang tiếng Anh
        metadata = {
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "doc_type": meta.get("loai_van_ban", ""),
            "authority": meta.get("co_quan_ban_hanh", ""),
            "issue_date": meta.get("ngay_ban_hanh", ""),
            "effective_date": meta.get("ngay_co_hieu_luc", ""),
            "status": meta.get("tinh_trang_hieu_luc", "")
        }
        
        # Thêm các metadata khác nếu có
        for k, v in meta.items():
            if k not in ["title", "loai_van_ban", "co_quan_ban_hanh", "ngay_ban_hanh", "ngay_co_hieu_luc", "tinh_trang_hieu_luc"]:
                metadata[k] = v
                
        doc = Document(
            page_content=row.get("content_html", ""),
            metadata=metadata
        )
        docs.append(doc)
        
    return docs
