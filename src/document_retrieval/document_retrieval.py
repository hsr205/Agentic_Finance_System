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

    async def retrieve_multiple_documents(self, tickers_list: list[str]) -> dict:
        # Create a list of concurrent async task_list
        task_list: list = [self._retrieve_document(ticker) for ticker in tickers_list]

        # Run them all at the same time and wait for result_list
        result_list: list = await asyncio.gather(*task_list)

        # Convert list of tuples into a dictionary mapping ticker -> 10-K text
        return dict(result_list)

    async def _retrieve_document(self, ticker_str: str) -> str:
        def _fetch_document_text():
            company_obj: Company = Company(cik_or_ticker=ticker_str)
            entity_filings: EntityFilings = company_obj.get_filings(form="10-K")
            if len(entity_filings) > 0:
                return entity_filings[0].text()
            return "No 10-K found"

        text_str: str = await asyncio.to_thread(_fetch_document_text)
        return text_str
