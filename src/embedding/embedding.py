from logging import Logger

from sentence_transformers import SentenceTransformer

from config.config import Settings
from huggingface_hub import login

from logger.logger import AppLogger


class Embedding:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        login(token=self._settings.hugging_face_token)
        self._model = SentenceTransformer(model_name_or_path=self._settings.embedding_model)
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    # TODO: Add parameter 'documents' that passes the chunked documents
    def embed_documents(self) -> None:
        sentences = ["Hugging Face makes NLP easy.", "Sentence transformers are powerful."]

        # 3. Calculate embeddings
        embeddings = self._model.encode(sentences)

        self._logger.info(f"Embedding shape: {embeddings.shape}")
