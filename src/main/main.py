from logging import Logger

from config.config import Settings
from embedding.embedding import Embedding
from logger.logger import AppLogger


def main() -> int:

    settings:Settings = Settings()
    embedding:Embedding = Embedding(settings=settings)
    logger: Logger = AppLogger().get_logger(class_name=str(__name__))

    try:
        logger.info(f"Inside {__name__}")

        embedding.embed_documents()

        return 0
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()

