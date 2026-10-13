import asyncio
from logging import Logger

from config.config import Settings
from document_retrieval.document_retrieval import DocumentRetrieval
from logger.logger import AppLogger


async def main() -> int:
    settings: Settings = Settings()
    # embedding: Embedding = Embedding(settings=settings)
    document_retrieval: DocumentRetrieval = DocumentRetrieval()
    logger: Logger = AppLogger().get_logger(class_name=str(__name__))

    try:
        logger.info(f"Inside {__name__}")

        # embedding.embed_documents()
        await document_retrieval.retrieve_documents()

        return 0
    except Exception as e:
        raise e


if __name__ == "__main__":
    asyncio.run(main())
