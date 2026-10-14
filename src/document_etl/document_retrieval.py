import asyncio
from logging import Logger

from edgar import set_identity, Company
from edgar.entity import EntityFilings

from config.config import Settings
from logger.logger import AppLogger


class DocumentRetrieval:
    def __init__(self, settings: Settings):
        self._settings = settings
        set_identity(user_identity=self._settings.edgar_identity_credentials)
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    async def retrieve_multiple_documents(self, tickers_list: list[str]) -> dict[str, str]:
        # Create a list of concurrent async task_list
        task_list: list = [self._retrieve_document(ticker) for ticker in tickers_list]

        # Run them all at the same time and wait for document_list
        document_list = await asyncio.gather(*task_list)

        self._logger.info(f"Retrieved {len(document_list):,} documents")
        self._logger.info("=" * 100)

        result_dict: dict[str, str] = self._get_document_dict(document_list=document_list)

        return result_dict

    def _get_document_dict(self, document_list) -> dict[str, str]:
        result_dict: dict[str, str] = {}

        for index, document in enumerate(document_list):
            document_key_str: str = document.get("metadata_dict").get("ticker") + "_" + document.get(
                "metadata_dict").get("form") + "_" + str(document.get("metadata_dict").get("filing_date"))
            document_text_str: str = document.get("text")
            result_dict[document_key_str] = document_text_str

        return result_dict

    async def _retrieve_document(self, ticker_str: str) -> dict:
        def _fetch_document_text():
            company_obj: Company = Company(cik_or_ticker=ticker_str)
            entity_filings: EntityFilings = company_obj.get_filings(form="10-K")
            latest_10k = entity_filings[0]

            # Extract metadata_dict attributes directly from the filing object
            metadata_dict: dict = {
                "ticker": ticker_str,
                "form": latest_10k.form,
                "filing_date": latest_10k.filing_date,
            }

            # Grab the text for your NLP/LLM pipeline
            text_str: str = latest_10k.text()

            return {
                "metadata_dict": metadata_dict,
                "text": text_str
            }
            return "No 10-K found"

        result_dict: dict = await asyncio.to_thread(_fetch_document_text)
        return result_dict
