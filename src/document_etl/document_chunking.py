from logging import Logger

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config.config import Settings
from document_etl.document_retrieval import DocumentRetrieval
from logger.logger import AppLogger


class DocumentChunking:

    def __init__(self, settings: Settings, document_retrieval:DocumentRetrieval) -> None:
        self._settings = settings
        self._document_retrieval: DocumentRetrieval = document_retrieval
        self._text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.rag_system_chunk_size,  # The maximum number of characters in each chunk
            chunk_overlap=self._settings.rag_system_chunk_overlap,
            # Number of characters to overlap between adjacent chunks
            length_function=len  # Function used to calculate chunk length (default is len)
        )
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    async def get_chunked_documents_dict(self) -> dict[str, list[Document]]:
        # TODO: Tickers_List will not be statically passed
        retrieved_documents_dict: dict = await self._document_retrieval.retrieve_multiple_documents(
            tickers_list=["AAPL", "GOOGL"])

        chunked_documents_result_dict: dict[str, list[Document]] = {}

        for document_key, document_text in tqdm(retrieved_documents_dict.items(), desc="Chunking Documents:"):
            document_obj: list[Document] = self._text_splitter.create_documents([document_text])
            chunked_documents_result_dict[document_key] = document_obj

        return chunked_documents_result_dict