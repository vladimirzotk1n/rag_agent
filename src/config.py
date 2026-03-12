from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    collection_name: str
    dense_model_name: str
    chunks_path: str = "/data/json/queried_tk_rf_flat.json"
    sparce_model_name: str = "Qdrant/bm25"
    use_cuda: bool = False
    dense_embedding_dim: int = 384
    hf_token: str
    e5_model: str
    tokenizer: str
    llm_model: str
    redis_url: str

    langsmith_tracing: bool
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str

    openai_api_key: str
    redis_password: str
    redis_user: str
    redis_user_password: str

    multihop: bool
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
