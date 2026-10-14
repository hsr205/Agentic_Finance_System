import asyncio
from logging import Logger

from config.config import Settings
from document_etl.document_chunking import DocumentChunking
from document_etl.document_retrieval import DocumentRetrieval
from logger.logger import AppLogger
from vector_store.vector_store import VectorStore


async def main() -> int:
    settings: Settings = Settings()
    # embedding: Embedding = Embedding(settings=settings)
    # document_etl: DocumentRetrieval = DocumentRetrieval(settings=settings)
    # vector_store: VectorStore = VectorStore()
    document_chunking: DocumentChunking = DocumentChunking(settings=settings)
    logger: Logger = AppLogger().get_logger(class_name=str(__name__))

    logger.info("Executing Application")
    logger.info("=" * 100)

    try:
        # embedding.embed_documents()
        # await document_etl.retrieve_documents()
        # vector_store.search_collection()
        await document_chunking.chunk_documents()

        logger.info("Successfully Completed Application Execution")
        logger.info("=" * 100)
        return 0
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    asyncio.run(main())
