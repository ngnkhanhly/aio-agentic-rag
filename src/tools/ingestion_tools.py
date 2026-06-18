import json
import os
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from langchain_core.tools import tool
from langchain_core.documents import Document
from configs.setting import config, HF_TOKEN

DATA_PROCESSED_DIR = Path("./data_processed")

def get_data_processed_dir() -> Path:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_PROCESSED_DIR

@tool
def load_dataset_tool(sample_size: int = 0) -> str:
    """
    Download the Vietnamese legal documents dataset from HuggingFace, map metadata,
    and save raw documents to data_processed/raw_docs.json.

    Args:
        sample_size: Number of documents to load (0 = use full dataset).
    Returns:
        Summary string.
    """
    from src.ingestion.loader import load_documents

    print("Downloading dataset from HuggingFace...")
    try:
        content_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/content.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df_content = pd.read_parquet(content_path)
        
        metadata_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/metadata.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df_metadata = pd.read_parquet(metadata_path)
    except Exception as e:
        return f"Error downloading dataset: {str(e)}"

    df_content = df_content.fillna("")
    df_metadata = df_metadata.fillna("")

    df_content['id'] = df_content['id'].astype(str)
    dataset_rows = df_content.to_dict(orient="records")

    df_metadata['id'] = df_metadata['id'].astype(str)
    meta_lookup = df_metadata.set_index('id').to_dict(orient="index")

    limit = sample_size if sample_size > 0 else None
    docs = load_documents(dataset_rows, meta_lookup, sample_size=limit)

    out_path = get_data_processed_dir() / "raw_docs.json"
    serialized = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    out_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return f"Successfully loaded {len(docs)} documents and saved to {out_path}."

@tool
def load_relationships_tool() -> str:
    """
    Download cross-document relationships from HuggingFace, filter them based on raw_docs.json
    if it exists, and save to data_processed/relationships.json.
    Returns:
        Summary string.
    """
    print("Downloading relationships from HuggingFace...")
    try:
        relationships_path = hf_hub_download(
            repo_id="th1nhng0/vietnamese-legal-documents",
            filename="data/relationships.parquet",
            repo_type="dataset",
            token=HF_TOKEN
        )
        df_relationships = pd.read_parquet(relationships_path)
    except Exception as e:
        return f"Error downloading relationships: {str(e)}"

    df_relationships = df_relationships.fillna("")
    df_relationships['doc_id'] = df_relationships['doc_id'].astype(str)
    df_relationships['other_doc_id'] = df_relationships['other_doc_id'].astype(str)
    relationships_list = df_relationships.to_dict(orient="records")

    # Filter based on loaded docs if possible
    raw_path = get_data_processed_dir() / "raw_docs.json"
    if raw_path.exists():
        try:
            with open(raw_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded_ids = {d["metadata"].get("doc_id", "") for d in data}
            relationships_list = [
                r for r in relationships_list
                if r.get("doc_id") in loaded_ids or r.get("other_doc_id") in loaded_ids
            ]
        except Exception as e:
            print(f"Warning: Failed to filter relationships based on raw_docs: {e}")

    out_path = get_data_processed_dir() / "relationships.json"
    out_path.write_text(json.dumps(relationships_list, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"Successfully loaded {len(relationships_list)} relationships and saved to {out_path}."

@tool
def clean_docs_tool() -> str:
    """
    Clean HTML and normalize text in all raw documents.
    Reads from raw_docs.json and saves to data_processed/cleaned_docs.json.
    Returns:
        Summary string.
    """
    from src.ingestion.cleaner import clean_documents

    raw_path = get_data_processed_dir() / "raw_docs.json"
    if not raw_path.exists():
        return f"Error: {raw_path} does not exist. Run load_dataset_tool first."

    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    except Exception as e:
        return f"Error reading raw_docs: {str(e)}"

    cleaned = clean_documents(docs, workers=1)
    
    out_path = get_data_processed_dir() / "cleaned_docs.json"
    serialized = [{"page_content": d.page_content, "metadata": d.metadata} for d in cleaned]
    out_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"Successfully cleaned {len(cleaned)} documents and saved to {out_path}."

@tool
def chunk_docs_tool() -> str:
    """
    Split cleaned documents into chunks using separator lists optimized for legal documents.
    Reads from cleaned_docs.json and saves to data_processed/chunks.json.
    Returns:
        Summary string.
    """
    from src.ingestion.chunker import chunk_documents

    cleaned_path = get_data_processed_dir() / "cleaned_docs.json"
    if not cleaned_path.exists():
        return f"Error: {cleaned_path} does not exist. Run clean_docs_tool first."

    try:
        with open(cleaned_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    except Exception as e:
        return f"Error reading cleaned_docs: {str(e)}"

    chunks = chunk_documents(docs, config)

    out_path = get_data_processed_dir() / "chunks.json"
    serialized = [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks]
    out_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")

    return f"Successfully split {len(docs)} documents into {len(chunks)} chunks, saved to {out_path}."

@tool
def build_chroma_tool() -> str:
    """
    Embed all chunks and insert them into local Chroma vector database.
    Reads from chunks.json.
    Returns:
        Summary string.
    """
    from src.indexing.chroma_store import get_vector_store

    chunks_path = get_data_processed_dir() / "chunks.json"
    if not chunks_path.exists():
        return f"Error: {chunks_path} does not exist. Run chunk_docs_tool first."

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    except Exception as e:
        return f"Error reading chunks: {str(e)}"

    store = get_vector_store()
    
    batch_size = 500
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        store.add_documents(batch)
        print(f"Added {i + len(batch)}/{len(chunks)} chunks to ChromaDB.")

    return f"Successfully indexed {len(chunks)} chunks into ChromaDB vector store."

@tool
def build_bm25_tool() -> str:
    """
    Build BM25 keyword index over all chunks and save to bm25_index.pkl.
    Reads from chunks.json.
    Returns:
        Summary string.
    """
    from src.indexing.bm25_index import build_bm25_index

    chunks_path = get_data_processed_dir() / "chunks.json"
    if not chunks_path.exists():
        return f"Error: {chunks_path} does not exist. Run chunk_docs_tool first."

    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
    except Exception as e:
        return f"Error reading chunks: {str(e)}"

    build_bm25_index(chunks)
    return "Successfully built and saved BM25 index."

@tool
def build_graph_tool() -> str:
    """
    Build cross-document relationship graph and save to graph.pkl.
    Reads from relationships.json.
    Returns:
        Summary string.
    """
    from src.retrieval.graph import build_graph, save_graph

    rels_path = get_data_processed_dir() / "relationships.json"
    if not rels_path.exists():
        return f"Error: {rels_path} does not exist. Run load_relationships_tool first."

    try:
        with open(rels_path, "r", encoding="utf-8") as f:
            relationships_list = json.load(f)
    except Exception as e:
        return f"Error reading relationships: {str(e)}"

    G = build_graph(relationships_list)
    save_graph(G)

    return f"Successfully built relationship graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges, saved to graph.pkl."
