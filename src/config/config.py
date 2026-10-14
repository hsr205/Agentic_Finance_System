from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str = Field(..., description="Sentence transformer embedding model for converting documents into embedded vectors")
    hugging_face_token: str = Field(..., description="Secret token leveraged for accessing Hugging Face resources")

    edgar_identity_credentials: str = Field(..., description="Credentials required for access the SEC EDGAR repository which house all financial documents (i.e. 10-k's, 8-k's, etc)")

    rag_system_chunk_size: int = Field(..., description="The maximum number of characters in each chunk")
    rag_system_chunk_overlap: int = Field(..., description="Number of characters to overlap between adjacent chunks")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
