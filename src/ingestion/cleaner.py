import unicodedata
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def _strip_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def _normalize(text: str) -> str:
    if not text:
        return ""
    # Chuẩn hóa NFC
    text = unicodedata.normalize("NFC", text)
    # Có thể thêm các bước chuẩn hóa khoảng trắng thừa nếu cần
    text = " ".join(text.split())
    return text

def clean_document(doc: Document) -> Document:
    raw = doc.page_content
    # Chỉ parse HTML nếu chuỗi chứa thẻ
    if "<" in raw and ">" in raw:
        raw = _strip_html(raw)
    
    cleaned = _normalize(raw)
    return Document(page_content=cleaned, metadata=doc.metadata)

def clean_documents(docs: list[Document], workers: int = 1) -> list[Document]:
    if workers <= 1:
        return [clean_document(d) for d in tqdm(docs, desc="Cleaning", unit="doc")]
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(
            executor.map(clean_document, docs, chunksize=64),
            total=len(docs),
            desc="Cleaning",
            unit="doc"
        ))
    return results
