import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import CandidateStore
from rrl.ingest import Ingester
from rrl.retriever import Retriever
from rrl.integrations.llama_index import RRLLlamaIndexRetriever
from rrl.feedback import OutcomeSignals

# 1. Initialize candidate store & ingester
store = CandidateStore()
ingester = Ingester()

# 2. Ingest some documents
doc_id = "doc_2"
text = "Thompson Sampling in RRL selects candidates based on a robust Beta distribution estimator."
ingester.ingest_document(store, doc_id, text)

# 3. Create Retriever and wrap it in LlamaIndex adapter
retriever = Retriever(store)
li_retriever = RRLLlamaIndexRetriever(rrl_retriever=retriever)

# 4. Perform retrieval using LlamaIndex API
print("Retrieving...")
nodes = li_retriever.retrieve("How does RRL explore?")
for node in nodes:
    print(
        f"Retrieved Node: {node.node.text} | Score: {node.score} | Metadata: {node.node.metadata}"
    )

# 5. Downstream outcome is observed (e.g. successful run)
print("Recording feedback...")
signals = OutcomeSignals(s_gt=1.0)
li_retriever.record_feedback(nodes, signals)

print("Check store candidate counters after feedback:")
cand = store.get_candidate("doc_2_chunk_0")
print(f"Candidate alpha: {cand.alpha}, beta: {cand.beta}")
