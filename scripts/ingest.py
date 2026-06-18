import sys
import os
import pandas as pd
from huggingface_hub import hf_hub_download

# Add root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from configs.setting import config, HF_TOKEN
from src.ingestion.loader import load_documents
from src.ingestion.cleaner import clean_documents
from src.ingestion.chunker import chunk_documents
from src.indexing.chroma_store import get_vector_store
from src.indexing.bm25_index import build_bm25_index
from src.retrieval.graph import build_graph, save_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Số lượng mẫu để test")
    args = parser.parse_args()

    print("1. Tải và đọc dữ liệu từ HuggingFace...")
    try:
        print("- Tải content.parquet...")
        content_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/content.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df_content = pd.read_parquet(content_path)
        
        print("- Tải metadata.parquet...")
        metadata_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/metadata.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df_metadata = pd.read_parquet(metadata_path)
        
        print("- Tải relationships.parquet...")
        relationships_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/relationships.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df_relationships = pd.read_parquet(relationships_path)
    except Exception as e:
        print("Lỗi khi tải hoặc đọc file parquet:", e)
        sys.exit(1)

    # Chuyển đổi dữ liệu sang format loader có thể xử lý
    # Điền chuỗi rỗng cho các giá trị NaN để tránh lỗi JSON serialization
    df_content = df_content.fillna("")
    df_metadata = df_metadata.fillna("")
    df_relationships = df_relationships.fillna("")

    df_content['id'] = df_content['id'].astype(str)
    dataset_rows = df_content.to_dict(orient="records")

    # meta_lookup: dict mapping id -> metadata dict
    df_metadata['id'] = df_metadata['id'].astype(str)
    meta_lookup = df_metadata.set_index('id').to_dict(orient="index")

    # Tải documents
    docs = load_documents(dataset_rows, meta_lookup, sample_size=args.sample)
    print(f"Đã tải {len(docs)} documents.")

    print("2. Làm sạch văn bản...")
    cleaned_docs = clean_documents(docs, workers=4)

    print("3. Chia chunk...")
    chunks = chunk_documents(cleaned_docs, config)
    print(f"Tạo được {len(chunks)} chunks.")

    print("4. Xây dựng các chỉ mục...")
    print("- Xây dựng BM25 Index...")
    build_bm25_index(chunks)
    
    print("- Xây dựng quan hệ đồ thị (Graph Index)...")
    # Chuyển relationships sang list of dicts
    df_relationships['doc_id'] = df_relationships['doc_id'].astype(str)
    df_relationships['other_doc_id'] = df_relationships['other_doc_id'].astype(str)
    relationships_list = df_relationships.to_dict(orient="records")
    G = build_graph(relationships_list)
    save_graph(G)
    print(f"  Đã lưu đồ thị với {G.number_of_nodes()} nodes và {G.number_of_edges()} edges.")

    print("- Xây dựng Vector Index (ChromaDB)...")
    store = get_vector_store()
    
    # Batch add to Chroma
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        store.add_documents(batch)
        print(f"  Đã thêm {i + len(batch)}/{len(chunks)} chunks vào ChromaDB")

    print("Hoàn tất Ingestion Pipeline!")

if __name__ == "__main__":
    main()
