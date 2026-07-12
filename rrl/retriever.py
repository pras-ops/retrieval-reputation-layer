"""
RRL Retriever Implementation
Implements hybrid retrieval (vector + BM25 RRF-fused), normalized sim scores,
and combines them with short-term (Beta sampled or expectation) and permanent usefulness.
Supports both real text queries (via SentenceTransformer + BM25) and pre-computed similarity scores.
"""

from collections import Counter
import math
import random
import re
from typing import Dict, List, Tuple, Optional, Union, Any

from .store import Candidate, CandidateStore
from .clustering import QueryClusterer

# Lazy import sentence-transformers only inside the model property to avoid heavy startup dependency.


def tokenize(text: str) -> List[str]:
    """Cleans punctuation, lowercases, and splits text into tokens."""
    return re.findall(r"\w+", text.lower())


class BM25:
    def __init__(self, candidates: List[Candidate], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.candidates = candidates
        self.corpus_size = len(candidates)

        # Tokenize doc contents
        self.doc_tokens = [tokenize(c.content) for c in candidates]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.corpus_size if self.corpus_size > 0 else 1.0

        self.doc_tfs = [Counter(tokens) for tokens in self.doc_tokens]

        # Document frequencies (df)
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

        # Precompute IDF
        self.idf: Dict[str, float] = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        scores = {}
        for i, candidate in enumerate(self.candidates):
            score = 0.0
            tf = self.doc_tfs[i]
            doc_len = self.doc_lens[i]
            for token in query_tokens:
                if token not in tf:
                    continue
                f = tf[token]
                idf_val = self.idf.get(token, 0.0)
                denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf_val * (f * (self.k1 + 1.0)) / denom
            scores[candidate.id] = score
        return scores


class Retriever:
    def __init__(
        self,
        store: CandidateStore,
        k_rrf: int = 60,
        weights: Tuple[float, float, float, float] = (0.70, 0.20, 0.10, 0.0),
        model_name: str = "all-MiniLM-L6-v2",
        model: Optional[Any] = None,
        robust_estimator_mode: str = "beta",
        use_optimistic_prior: bool = True,
        clusterer: Optional[QueryClusterer] = None,
        use_clustering: bool = False,
    ):
        self.store = store
        self.k_rrf = k_rrf
        self.weights = weights
        self.model_name = model_name
        self._model = model
        self.robust_estimator_mode = robust_estimator_mode
        self.use_optimistic_prior = use_optimistic_prior
        self.use_clustering = use_clustering

        self.clusterer: QueryClusterer
        if clusterer is None:
            self.clusterer = QueryClusterer()
            self.clusterer.load(self.store)
        else:
            self.clusterer = clusterer

        # BM25 index cache
        self._bm25_cached: Optional[BM25] = None
        self._bm25_candidate_ids: List[str] = []

        # Instantiate the reputation layer
        from .layer import ReputationLayer
        self.layer = ReputationLayer(
            store=self.store,
            weights=self.weights,
            robust_estimator_mode=self.robust_estimator_mode,
            use_optimistic_prior=self.use_optimistic_prior,
            gamma=1.0,
            decay_unit_sec=86400.0,
            use_clustering=self.use_clustering,
        )
        self.last_response_id: str = ""

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it using `pip install retrieval-reputation-layer[embeddings]` "
                    "or pass a custom embedder model instance to Retriever."
                )
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _compute_rrf(
        self, vector_scores: Dict[str, float], bm25_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Computes Reciprocal Rank Fusion (RRF) scores for candidates.
        """
        sorted_vector = [
            cid for cid, _ in sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        sorted_bm25 = [
            cid for cid, _ in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        vector_ranks = {cid: rank + 1 for rank, cid in enumerate(sorted_vector)}
        bm25_ranks = {cid: rank + 1 for rank, cid in enumerate(sorted_bm25)}

        all_cids = set(vector_scores.keys()).union(set(bm25_scores.keys()))
        rrf_scores = {}

        for cid in all_cids:
            v_rank = vector_ranks.get(cid, len(sorted_vector) + 1)
            b_rank = bm25_ranks.get(cid, len(sorted_bm25) + 1)

            v_part = 1.0 / (self.k_rrf + v_rank)
            b_part = 1.0 / (self.k_rrf + b_rank)

            rrf_scores[cid] = v_part + b_part

        return rrf_scores

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Min-max normalizes scores to the [0, 1] range.
        """
        if not scores:
            return {}

        vals = list(scores.values())
        min_val = min(vals)
        max_val = max(vals)
        diff = max_val - min_val

        if diff == 0:
            return {cid: 1.0 for cid in scores}

        return {cid: (val - min_val) / diff for cid, val in scores.items()}

    def retrieve(
        self,
        vector_scores: Union[str, Dict[str, float]],
        bm25_scores: Optional[Dict[str, float]] = None,
        top_k: int = 5,
        explore: bool = True,
        override_weights: Optional[Tuple[float, float, float, float]] = None,
        epsilon: float = 0.15,
        robust_estimator_mode: Optional[str] = None,
        current_timestamp: Optional[float] = None,
        gamma: float = 1.0,
        decay_unit_sec: float = 86400.0,
    ) -> List[Tuple[Candidate, float, float]]:
        """
        Retrieves the top_k candidates.

        Parameters:
            vector_scores: Either a real text query (str) or pre-computed vector similarity dict.
            bm25_scores: (Optional) Pre-computed BM25 similarity dict (only if first param is a dict).
            top_k: Number of candidates to retrieve.
            explore: Whether to apply Beta distribution sampling and rarity bonus.
            override_weights: Custom weights (w_sim, w_c, w_p, w_explore).
            epsilon: Epsilon-greedy parameter.
        """
        w_sim, w_c, w_p, w_explore = (
            override_weights if override_weights is not None else self.weights
        )

        self.last_query_cluster = "cluster_0" if self.use_clustering else None
        cluster_id = "cluster_0" if self.use_clustering else None

        # Check if first parameter is a real text query
        if isinstance(vector_scores, str):
            query = vector_scores
            candidates = self.store.list_candidates()
            if not candidates:
                return []

            # 1. Compute query embedding
            query_emb = self.model.encode(query).tolist()
            if self.use_clustering:
                cluster_id = self.clusterer.assign(query_emb)
                self.clusterer.save(self.store)
                self.last_query_cluster = cluster_id
            else:
                cluster_id = None
                self.last_query_cluster = None
            query_tokens = tokenize(query)

            # 2. Calculate vector similarity (dot product of unit length vectors)
            vector_scores = {}
            for cand in candidates:
                cand_emb = cand.metadata.get("embedding")
                if cand_emb is None:
                    # Dynamically encode and cache if missing
                    cand_emb = self.model.encode(cand.content).tolist()
                    cand.metadata["embedding"] = cand_emb

                # Cosine similarity
                dot_product = sum(q * c for q, c in zip(query_emb, cand_emb))
                norm_q = sum(q * q for q in query_emb) ** 0.5
                norm_c = sum(c * c for c in cand_emb) ** 0.5
                vector_scores[cand.id] = (
                    dot_product / (norm_q * norm_c) if (norm_q * norm_c) > 0 else 0.0
                )

            # 3. Calculate BM25 scores (cached or rebuilt)
            current_cids = sorted(cand.id for cand in candidates)
            if self._bm25_cached is None or self._bm25_candidate_ids != current_cids:
                self._bm25_cached = BM25(candidates)
                self._bm25_candidate_ids = current_cids

            calculated_bm25_scores = self._bm25_cached.get_scores(query_tokens)
        else:
            # Pre-computed scores (backward compatible for simulation)
            calculated_bm25_scores = bm25_scores if bm25_scores is not None else {}

        # 4. Compute RRF fused scores
        rrf_scores = self._compute_rrf(vector_scores, calculated_bm25_scores)

        # 5. Normalize RRF scores to obtain sim(i) in [0, 1]
        sim_scores = self._normalize_scores(rrf_scores)

        # Delegate ranking, scoring and epsilon-greedy exploration to ReputationLayer
        rescore_result = self.layer.rescore(
            sims=sim_scores,
            top_k=top_k,
            explore=explore,
            now=current_timestamp,
            cluster_id=cluster_id,
            override_weights=override_weights,
            epsilon=epsilon,
            robust_estimator_mode=robust_estimator_mode,
            gamma=gamma,
            decay_unit_sec=decay_unit_sec,
        )
        self.last_response_id = rescore_result.response_id

        # Resolve IDs back to Candidate tuples for backward compatibility
        results = []
        for cid, score, sim in rescore_result.results:
            cand = self.store.get_candidate(cid)
            if cand:
                results.append((cand, score, sim))

        return results
