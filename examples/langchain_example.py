import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rrl.store import CandidateStore
from rrl.ingest import Ingester
from rrl.retriever import Retriever
from rrl.integrations.langchain import RRLRetriever
from rrl.feedback import OutcomeSignals

# 1. Initialize candidate store & ingester
store = CandidateStore()
ingester = Ingester()

# 2. Ingest some documents
doc_id = "doc_1"
text = "RRL (Retrieval Reputation Layer) is a feedback-driven reranking layer."
ingester.ingest_document(store, doc_id, text)

# 3. Create Retriever and wrap it in LangChain adapter
retriever = Retriever(store)
lc_retriever = RRLRetriever(rrl_retriever=retriever)

# 4. Perform retrieval using the LangChain API
print("Retrieving...")
docs = lc_retriever.invoke("What is RRL?")
for doc in docs:
    print(f"Retrieved: {doc.page_content} | Metadata: {doc.metadata}")

# 5. Downstream execution outcome is observed (e.g., successful run)
print("Recording feedback...")
signals = OutcomeSignals(s_gt=1.0)  # ground truth successful
lc_retriever.record_feedback(docs, signals)

print("Check store candidate counters after feedback:")
cand = store.get_candidate("doc_1_chunk_0")
print(f"Candidate alpha: {cand.alpha}, beta: {cand.beta}")
