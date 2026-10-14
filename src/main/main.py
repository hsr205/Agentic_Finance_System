import asyncio
from logging import Logger

from config.config import Settings
from document_etl.document_chunking import DocumentChunking
from document_etl.document_retrieval import DocumentRetrieval
from embedding.embedding import Embedding
from logger.logger import AppLogger
from vector_store.vector_store import VectorStore


async def main() -> int:
    settings: Settings = Settings()
    document_retrieval: DocumentRetrieval = DocumentRetrieval(settings=settings)
    document_chunking: DocumentChunking = DocumentChunking(settings=settings, document_retrieval=document_retrieval)
    embedding: Embedding = Embedding(settings=settings, document_chunking=document_chunking)
    vector_store: VectorStore = VectorStore(settings=settings, embedding=embedding, document_chunking=document_chunking)

    logger: Logger = AppLogger().get_logger(class_name=str(__name__))

    logger.info("Executing Application")
    logger.info("=" * 100)

    try:
        # await embedding.get_embedded_documents_dict()
        # await document_etl.retrieve_documents()
        # vector_store.search_collection()
        # await document_chunking.get_chunked_documents_dict()
        # await vector_store.create_qdrant_collection()

        query_text: str = "What was the total net income?"
        vector_store.search_collection(query_text=query_text)

        logger.info("Successfully Completed Application Execution")
        logger.info("=" * 100)
        return 0
    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    asyncio.run(main())
