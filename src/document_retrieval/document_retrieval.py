import asyncio
from logging import Logger

from edgar import set_identity, Company
from edgar.entity import EntityFilings, EntityFiling

from config.config import Settings
from logger.logger import AppLogger


class DocumentRetrieval:
    def __init__(self, settings: Settings):
        self._settings = settings
        set_identity(user_identity=self._settings.edgar_identity_credentials)
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    async def retrieve_documents(self) -> str:
        text_str: str = await asyncio.to_thread(self._fetch)
        return text_str

    # TODO: Alter this method to output a list of documents not just a single filling
    #       Additionally this method should retrieve multiple filings from multiple publicly traded companies
    def _fetch(self):
        company_obj: Company = Company(cik_or_ticker="AAPL")
        entity_filings_obj: EntityFilings = company_obj.get_filings(form="10-K")
        latest_10k: EntityFiling = entity_filings_obj[0]
        return latest_10k.text()
