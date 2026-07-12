from typing import Any, List

from rrl.feedback import OutcomeSignals

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:
    raise ImportError(
        "langchain-core is not installed. Install it using "
        "`pip install retrieval-reputation-layer[langchain]`."
    ) from e


class RRLRetriever(BaseRetriever):
    """
    RRL LangChain Retriever Adapter.
    Wraps the core RRL Retriever to output standard LangChain Documents.
    """

    rrl_retriever: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Retrieve candidates from core RRL retriever
        results = self.rrl_retriever.retrieve(query, explore=True)
        docs = []
        for cand, score, sim in results:
            docs.append(
                Document(
                    page_content=cand.content,
                    metadata={
                        "id": cand.id,
                        "rrl_sim": sim,
                        "rrl_score": score,
                        **cand.metadata,
                    },
                )
            )
        return docs

    def record_feedback(self, docs: List[Document], signals: OutcomeSignals, **kwargs: Any):
        """
        Statelessly records downstream outcome feedback using the similarity scores
        and candidate IDs cached in the document metadata.
        """
        retrieved_sims = {
            doc.metadata["id"]: doc.metadata["rrl_sim"]
            for doc in docs
            if "rrl_sim" in doc.metadata and "id" in doc.metadata
        }
        from rrl.feedback import calculate_outcome, update_counters

        y = calculate_outcome(signals)
        if y is None:
            y = 1.0
        update_counters(self.rrl_retriever.store, retrieved_sims, y, signals=signals, **kwargs)
