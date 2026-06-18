import os
import pickle
from langchain_community.retrievers import BM25Retriever

BM25_INDEX_PATH = "bm25_index.pkl"

def save_bm25_index(retriever: BM25Retriever, path: str = BM25_INDEX_PATH):
    """
    Lưu BM25 retriever ra file.
    """
    with open(path, "wb") as f:
        pickle.dump(retriever, f)

def load_bm25_index(path: str = BM25_INDEX_PATH) -> BM25Retriever:
    """
    Tải BM25 retriever từ file.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def build_bm25_index(docs) -> BM25Retriever:
    """
    Xây dựng chỉ mục BM25 từ danh sách Document.
    """
    retriever = BM25Retriever.from_documents(docs)
    save_bm25_index(retriever)
    return retriever
