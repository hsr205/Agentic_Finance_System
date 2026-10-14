from logging import Logger

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config.config import Settings
from document_etl.document_retrieval import DocumentRetrieval
from logger.logger import AppLogger


class DocumentChunking:

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._document_retrieval: DocumentRetrieval = DocumentRetrieval(self._settings)
        self._text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.rag_system_chunk_size,  # The maximum number of characters in each chunk
            chunk_overlap=self._settings.rag_system_chunk_overlap,
            # Number of characters to overlap between adjacent chunks
            length_function=len  # Function used to calculate chunk length (default is len)
        )
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    async def chunk_documents(self) -> None:
        # TODO: Tickers_List will not be statically passed
        retrieved_documents_dict: dict = await self._document_retrieval.retrieve_multiple_documents(
            tickers_list=["AAPL", "GOOGL"])

        self._logger.info(f"Chunking {len(retrieved_documents_dict)} documents")
        self._logger.info("=" * 100)

        for document_key, document_text in tqdm(retrieved_documents_dict.items(), desc="Chunking Documents:"):
            document_obj: list[Document] = self._text_splitter.create_documents([document_text])
            self._logger.info(f"Chunking {len(document_obj)} documents")
            self._logger.info(f"document_obj = {document_obj}")
            self._logger.info("=" * 100)
