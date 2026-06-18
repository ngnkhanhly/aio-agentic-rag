from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def build_splitter(config) -> RecursiveCharacterTextSplitter:
    """
    Tạo splitter với các separator ưu tiên cấu trúc văn bản pháp luật tiếng Việt.
    """
    return RecursiveCharacterTextSplitter(
        separators=[
            "\nĐiều ", "\nKhoản ", "\nĐiểm ",  # Ưu tiên ranh giới pháp luật
            "\n\n", "\n", " ", ""              # Ranh giới thông thường
        ],
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )

def chunk_documents(docs: list[Document], config) -> list[Document]:
    """
    Chia nhỏ danh sách các Document và gán chunk_index.
    """
    splitter = build_splitter(config)
    chunks = splitter.split_documents(docs)
    
    # Gán index cho từng chunk để dễ deduplicate sau này (vd: trong graph retrieval)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        
    return chunks
