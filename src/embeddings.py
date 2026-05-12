import os
from typing import Sequence

import numpy as np
from google import genai
from google.genai import types


class GoogleEmbedder:
    """Gemini embedding wrapper for document and query embeddings."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings.")

        self._client = genai.Client(api_key=self.api_key)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def _embed_batch(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        response = self._client.models.embed_content(
            model=self.model,
            contents=list(texts),
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.output_dimensionality,
            ),
        )

        vectors = [item.values for item in response.embeddings]
        if not vectors:
            return np.zeros((0, self.output_dimensionality), dtype=np.float32)

        matrix = np.asarray(vectors, dtype=np.float32)
        return self._l2_normalize(matrix).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.output_dimensionality), dtype=np.float32)

        chunks: list[np.ndarray] = []
        chunk_size = 100
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start : start + chunk_size]
            chunks.append(self._embed_batch(chunk, task_type="RETRIEVAL_DOCUMENT"))

        return np.vstack(chunks).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        matrix = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        if matrix.shape[0] == 0:
            return np.zeros((self.output_dimensionality,), dtype=np.float32)
        return matrix[0].astype(np.float32)
