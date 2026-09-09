"""Semantic search over the ingested 10-K chunks."""

import chromadb

from src.config import CHROMA_DIR, get_embed_model
from src.llm import LLM
from src.models import SearchHit


class FilingsIndex:
    def __init__(self):
        embed_config = get_embed_model()
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(embed_config.collection_name())

    def search(
        self,
        query: str,
        llm: LLM,
        k: int = 8,
        ticker: str | None = None,
        fiscal_year: int | None = None,
    ) -> list[SearchHit]:
        """Return up to k chunks, optionally filtered to one company and/or
        fiscal year."""
        where = {}
        if ticker and fiscal_year:
            where = {"$and": [{"ticker": ticker}, {"fiscal_year": fiscal_year}]}
        elif ticker:
            where = {"ticker": ticker}
        elif fiscal_year:
            where = {"fiscal_year": fiscal_year}

        query_vector = llm.embed([query], is_query=True).vectors[0]
        result = self.collection.query(
            query_embeddings=[query_vector], n_results=k, where=where or None
        )

        hits = []
        for text, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(SearchHit(text=text, distance=distance, **metadata))
        return hits
