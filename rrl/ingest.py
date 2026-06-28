"""
RRL Ingestion Module
Handles document chunking and vector embedding generation.
"""

from typing import List, Dict, Any, Optional
from .store import Candidate, CandidateStore

_HAS_SENTENCE_TRANSFORMERS = True
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Splits text into chunks using a character-level sliding window.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class Ingester:
    def __init__(self, model: Optional[Any] = None, model_name: str = "all-MiniLM-L6-v2"):
        if model is not None:
            self.model = model
        else:
            if not _HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it using `pip install retrieval-reputation-layer[embeddings]` "
                    "or pass a custom embedder model instance to Ingester."
                )
            self.model = SentenceTransformer(model_name)

    def ingest_document(
        self,
        store: CandidateStore,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Chunks the document text, encodes each chunk to a vector embedding,
        creates Candidate objects, and registers them in the CandidateStore.

        Returns:
            List of generated candidate IDs.
        """
        chunks = chunk_text(text)
        candidate_ids = []

        # Bulk encode chunks for performance
        if not chunks:
            return []

        embeddings = self.model.encode(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            candidate_id = f"{doc_id}_chunk_{i}"

            # Convert embedding to list to ensure portability
            emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)

            # Pack embedding list and other attributes in metadata
            cand_metadata = {
                "doc_id": doc_id,
                "chunk_idx": i,
                "embedding": emb_list,
                **(metadata or {}),
            }

            candidate = Candidate(
                id=candidate_id,
                content=chunk,
                metadata=cand_metadata,
                alpha=1.0,
                beta=1.0,
                A=1.0,
                B=1.0,
            )

            store.add_candidate(candidate)
            candidate_ids.append(candidate_id)

        return candidate_ids
