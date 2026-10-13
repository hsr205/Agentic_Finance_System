from logging import Logger

from qdrant_client import QdrantClient
from qdrant_client.grpc import UpdateResult
from qdrant_client.http.models import ScoredPoint
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from logger.logger import AppLogger


class VectorStore:

    def __init__(self) -> None:
        # TODO: Pass the local host URL as an ENV variable
        self._qdrant_client: QdrantClient = QdrantClient(url="http://localhost:6333")
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    def create_qdrant_collection(self) -> None:

        try:

            # TODO: Make this dynamically passed not static
            collection_name_str: str = "10k_collection"

            self._logger.info(f"Creating collection -> {collection_name_str}")
            self._logger.info("=" * 100)

            self._qdrant_client.create_collection(
                collection_name="10k_collection",
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )

            operation_result: UpdateResult = self._qdrant_client.upsert(
                collection_name="10k_collection",
                wait=True,
                points=[
                    PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
                    PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
                    PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
                    PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
                    PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
                    PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
                ],
            )
            self._logger.info(f"Successfully created collection -> {collection_name_str}")
            self._logger.info(f"Operation result -> {operation_result}")
            self._logger.info("=" * 100)

        except Exception as e:
            self._logger.error("Exception thrown while creating collection")
            raise Exception(e)

    def search_collection(self) -> None:

        search_result_list: list[ScoredPoint] = self._qdrant_client.query_points(
            collection_name="test_collection",
            query=[0.2, 0.1, 0.9, 0.7],
            with_payload=False,
            limit=3
        ).points

        self._logger.info(f"search_result_list = {search_result_list}")
        self._logger.info("=" * 100)
