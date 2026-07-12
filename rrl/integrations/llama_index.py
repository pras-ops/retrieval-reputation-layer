from typing import Any, List

from rrl.feedback import OutcomeSignals

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as e:
    raise ImportError(
        "llama-index-core is not installed. Install it using "
        "`pip install retrieval-reputation-layer[llamaindex]`."
    ) from e


class RRLLlamaIndexRetriever(BaseRetriever):
    """
    RRL LlamaIndex Retriever Adapter.
    Wraps the core RRL Retriever to output standard LlamaIndex NodeWithScore objects.
    """

    def __init__(self, rrl_retriever: Any, **kwargs: Any):
        self._rrl_retriever = rrl_retriever
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        results = self._rrl_retriever.retrieve(query_str, explore=True)
        nodes = []
        for cand, score, sim in results:
            node = TextNode(
                text=cand.content,
                id_=cand.id,
                metadata={
                    "rrl_sim": sim,
                    **cand.metadata,
                },
            )
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes

    def record_feedback(self, nodes: List[NodeWithScore], signals: OutcomeSignals, **kwargs: Any):
        """
        Statelessly records downstream outcome feedback using the similarity scores
        and candidate IDs cached in the node metadata.
        """
        retrieved_sims = {
            node.node.id_: node.node.metadata["rrl_sim"]
            for node in nodes
            if "rrl_sim" in node.node.metadata
        }
        from rrl.feedback import calculate_outcome, update_counters

        y = calculate_outcome(signals)
        if y is None:
            y = 1.0
        update_counters(self._rrl_retriever.store, retrieved_sims, y, signals=signals, **kwargs)
