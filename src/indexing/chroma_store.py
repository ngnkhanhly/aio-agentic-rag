import chromadb
from langchain_chroma import Chroma
from src.indexing.embeddings import get_embeddings
from configs.setting import CHROMA_HOST, CHROMA_PORT

def get_vector_store(collection_name: str = "legal_docs") -> Chroma:
    """
    Kết nối tới ChromaDB (sử dụng PersistentClient cục bộ tại thư mục ./chroma_db
    để không cần cài đặt Docker/Chroma server) và trả về đối tượng vector store của Langchain.
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    embeddings = get_embeddings()
    
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vector_store
