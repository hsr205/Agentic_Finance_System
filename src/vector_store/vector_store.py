from logging import Logger

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint, PointStruct, UpdateResult, Document
from qdrant_client.models import Distance, VectorParams

from config.config import Settings
from document_etl.document_chunking import DocumentChunking
from embedding.embedding import Embedding
from logger.logger import AppLogger


# TODO: Explicitly set device (i.e. CUDA, MPS, CPU)
class VectorStore:

    def __init__(self, settings: Settings, embedding: Embedding, document_chunking: DocumentChunking) -> None:
        self._settings: Settings = settings
        self._embedding: Embedding = embedding
        self._document_chunking: DocumentChunking = document_chunking
        # TODO: Instead of passing this as an ENV variable the following bash command should be run:
        #       BASH COMMAND: docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
        self._qdrant_client: QdrantClient = QdrantClient(url=self._settings.qdrant_vector_store_client)
        self._logger: Logger = AppLogger.get_logger(self.__class__.__name__)

    async def create_qdrant_collection(self) -> None:

        try:

            collection_name_str: str = "10k_collection"

            # 1. Fetch chunked documents and embeddings dictionaries
            chunked_docs_dict: dict[
                str, list[Document]] = await self._document_chunking.get_chunked_documents_dict()
            embedded_documents_dict: dict[str, np.ndarray] = await self._embedding.get_embedded_documents_dict()

            point_struct_list: list[PointStruct] = []
            point_id_counter = 1

            # 2. Iterate through each ticker and its corresponding chunks & vectors
            for ticker_str, vectors_array in embedded_documents_dict.items():
                document_list: list[Document] = chunked_docs_dict.get(ticker_str, [])

                # Ensure lengths match
                for chunk_index, vector in enumerate(vectors_array):
                    document: Document = document_list[chunk_index]

                    # Create the PointStruct (no trailing comma!)
                    point = PointStruct(
                        id=point_id_counter,  # Or use uuid.uuid4() for unique IDs
                        vector=vector.tolist(),  # Convert numpy array to standard Python list
                        payload={
                            "ticker": ticker_str,
                            "text": document.page_content,  # Crucial: store the raw text chunk
                            "chunk_index": chunk_index
                        }
                    )

                    point_struct_list.append(point)
                    point_id_counter += 1

            self._logger.info(f"Prepared {len(point_struct_list):,} points for upserting.")
            self._logger.info("=" * 100)

            # 3. Ensure collection exists (using dynamic dimension size from model)
            vector_size_int: int = self._embedding.get_sentence_transformer_embedding_dimension()

            # Optional: check if collection exists or recreate it
            self._qdrant_client.create_collection(
                collection_name=collection_name_str,
                vectors_config=VectorParams(size=vector_size_int, distance=Distance.COSINE),
            )

            # 4. Upsert the list of points directly to Qdrant
            operation_result: UpdateResult = self._qdrant_client.upsert(
                collection_name=collection_name_str,
                wait=True,
                points=point_struct_list,
            )

            self._logger.info(f"Successfully upserted points to -> {collection_name_str}")
            self._logger.info(f"Operation result -> {operation_result}")

        except Exception as e:
            self._logger.error(e)
            raise Exception(e)

    def search_collection(self, query_text: str, limit: int = 3) -> list[ScoredPoint]:
        self._logger.info(f"Searching collection for query: '{query_text}'")

        # 1. Convert the natural language text into a vector using your SentenceTransformer model
        # Note: .encode() returns a numpy array, convert it to a list for Qdrant
        query_vector_str = self._embedding.get_embedding_model().encode(query_text).tolist()

        # 2. Query Qdrant using the vector
        search_result_list: list[ScoredPoint] = self._qdrant_client.query_points(
            collection_name="10k_collection",
            query=query_vector_str,
            with_payload=True,  # MUST be True to get your text chunks and ticker metadata back!
            limit=limit
        ).points

        # 3. Log or inspect the retrieved text payloads
        for point in search_result_list:
            self._logger.info(f"Score: {point.score:.4f} | Ticker: {point.payload.get('ticker')}")
            self._logger.info(f"Text Snippet: {point.payload.get('text')[:200]}...")
            self._logger.info("-" * 50)

        return search_result_list
