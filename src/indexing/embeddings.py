from langchain_openai import OpenAIEmbeddings
from configs.setting import LLM_PROVIDER, EMBEDDING_BASE_URL

def get_embeddings():
    """
    Khởi tạo model embeddings dựa trên cấu hình.
    Trong dự án này sử dụng OpenAIEmbeddings.
    """
    if LLM_PROVIDER.lower() == "openai":
        # Có thể sử dụng base_url nếu dùng API tương thích OpenAI
        kwargs = {}
        if EMBEDDING_BASE_URL:
            kwargs["openai_api_base"] = EMBEDDING_BASE_URL
        return OpenAIEmbeddings(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")
