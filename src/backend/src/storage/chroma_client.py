import chromadb
from typing import List, Dict
import numpy as np
import os


class ChromaStorage:
    def __init__(
        self,
        collection_name: str = "audio_samples",
        persist_directory: str = "./chroma_db",
    ):
        self.collection_name = collection_name

        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = os.getenv("CHROMA_PORT", "8000")

        if chroma_host == "chromadb":
            self.client = chromadb.HttpClient(host=chroma_host, port=str(chroma_port))
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)

        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Music sample embeddings"},
            )
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise

    def add_samples(
        self, embeddings: List[np.ndarray], metadata: List[Dict], ids: List[str]
    ):
        embeddings_list = [emb.tolist() for emb in embeddings]

        self.collection.add(embeddings=embeddings_list, metadatas=metadata, ids=ids)  # type: ignore[arg-type]

    def search_similar(self, query_embedding: np.ndarray, n_results: int = 5) -> Dict:
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=n_results
        )
        return results  # type: ignore[return-value]

    def get_all_samples(self) -> Dict:
        return self.collection.get(include=["embeddings", "metadatas"])  # type: ignore[return-value]

    def count(self) -> int:
        return self.collection.count()
