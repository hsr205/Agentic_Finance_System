import asyncio
from logging import Logger

from config.config import Settings
from document_retrieval.document_retrieval import DocumentRetrieval
from logger.logger import AppLogger
from vector_store.vector_store import VectorStore


async def main() -> int:
    settings: Settings = Settings()
    # embedding: Embedding = Embedding(settings=settings)
    # document_retrieval: DocumentRetrieval = DocumentRetrieval(settings=settings)
    vector_store: VectorStore = VectorStore()
    logger: Logger = AppLogger().get_logger(class_name=str(__name__))

    try:
        # embedding.embed_documents()
        # await document_retrieval.retrieve_documents()
        vector_store.search_collection()

        return 0
    except Exception as e:
        raise e


if __name__ == "__main__":
    asyncio.run(main())
