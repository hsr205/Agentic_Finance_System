import asyncio
from logging import Logger

from edgar import set_identity, Company
from edgar.entity import EntityFilings, EntityFiling

from logger.logger import AppLogger


class DocumentRetrieval:
    def __init__(self):
        set_identity("HenryRothenberg hsr205@aol.com")
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    # TODO: Alter this method to output a list of documents not just a single filling
    #       Additionally this method should retrieve multiple filings from multiple publicly traded companies
    async def retrieve_documents(self) -> str:
        def _fetch():
            company_obj: Company = Company(cik_or_ticker="AAPL")
            entity_filings_obj: EntityFilings = company_obj.get_filings(form="10-K")
            latest_10k: EntityFiling = entity_filings_obj[0]
            return latest_10k.text()

        text_str: str = await asyncio.to_thread(_fetch)
        return text_str
