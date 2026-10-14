from logging import Logger

import numpy as np
from huggingface_hub import login
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config.config import Settings
from document_etl.document_chunking import DocumentChunking
from logger.logger import AppLogger


class Embedding:
    def __init__(self, settings: Settings, document_chunking: DocumentChunking) -> None:
        self._settings = settings
        login(token=self._settings.hugging_face_token)
        self._model = SentenceTransformer(model_name_or_path=self._settings.embedding_model)
        self._document_chunking: DocumentChunking = document_chunking
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    async def get_embedded_documents_dict(self) -> dict[str, np.ndarray]:

        embedded_documents_dict: dict[str, np.ndarray] = {}
        chunked_documents_result_dict: dict[
            str, list[Document]] = await self._document_chunking.get_chunked_documents_dict()

        for ticker_key_str, document_list_obj in chunked_documents_result_dict.items():
            self._logger.info(f"Embedding {len(document_list_obj):,} chunks for ticker: {ticker_key_str}")

            # 2. Extract the text content from each LangChain Document object
            texts_to_embed = [doc.page_content for doc in document_list_obj]

            if not texts_to_embed:
                self._logger.warning(f"No documents found for ticker: {ticker_key_str}")
                continue

            # 3. Generate embeddings using SentenceTransformer
            # (batching happens automatically under the hood in sentence-transformers)
            embeddings_array: np.ndarray = self._model.encode(texts_to_embed, show_progress_bar=False)

            self._logger.info(f"Ticker {ticker_key_str} embedding shape: {embeddings_array.shape}")
            self._logger.info("=" * 100)

            # 4. Store the resulting numpy array mapped to the ticker
            embedded_documents_dict[ticker_key_str] = embeddings_array

        return embedded_documents_dict

    def get_embedding_model(self) -> SentenceTransformer:
        return self._model

    def get_sentence_transformer_embedding_dimension(self) -> int:
        return self._model.get_embedding_dimension()
