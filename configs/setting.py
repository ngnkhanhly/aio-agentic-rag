import os
import yaml
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class RetrievalConfig(BaseModel):
    k: int
    graph_max_hops: int
    rrf_k: int

class GenerationConfig(BaseModel):
    temperature: float
    max_tokens: int

class Config(BaseModel):
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

def load_config(yaml_path: str = "configs/config.yaml") -> Config:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

config = load_config()

# Environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
HF_TOKEN = os.getenv("HF_TOKEN", "")
