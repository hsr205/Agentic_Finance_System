from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str = Field(..., description="Sentence transformer embedding model for converting documents into embedded vectors")
    hugging_face_token: str = Field(..., description="Secret token leveraged for accessing Hugging Face resources")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
