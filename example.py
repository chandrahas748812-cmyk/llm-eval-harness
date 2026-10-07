"""Head-to-head: BM25 vs dense vs hybrid vs hybrid+rerank."""

from src.corpus import DOCUMENTS, QUERIES
from src.search import BM25Retriever, DenseRetriever, HybridRetriever
from src.rerank import Reranker, DOC_TEXT
from src.metrics import precision_at_k

DOC_TEXT.update({d["id"]: d["text"] for d in DOCUMENTS})

TOP_K = 10
EVAL_K = 3

print("Loading models...")
try:
    dense = DenseRetriever(DOCUMENTS)
except Exception as e:
    print(f"(dense unavailable: {e} — dense/hybrid will use BM25 only)")
    dense = None

bm25 = BM25Retriever(DOCUMENTS)
hybrid = HybridRetriever(DOCUMENTS, dense=dense)
reranker = Reranker()

totals = {"bm25": [], "dense": [], "hybrid": [], "hybrid+rerank": []}

for query, relevant in QUERIES.items():
    b_ids = [d for d, _ in bm25.search(query, TOP_K)]
    d_ids = [d for d, _ in dense.search(query, TOP_K)] if dense else b_ids
    h_ids = [d for d, _ in hybrid.search(query, TOP_K, alpha=0.5)]
    r_ids = [d for d, _ in reranker.rerank(query, h_ids, top_k=EVAL_K)]

    scores = {
        "bm25": precision_at_k(b_ids, relevant, EVAL_K),
        "dense": precision_at_k(d_ids, relevant, EVAL_K),
        "hybrid": precision_at_k(h_ids, relevant, EVAL_K),
        "hybrid+rerank": precision_at_k(r_ids, relevant, EVAL_K),
    }
    for k, v in scores.items():
        totals[k].append(v)

    best = max(scores, key=scores.get)
    print(f'\nQuery: "{query}"')
    for name, s in scores.items():
        mark = "  ✅" if name == best else ""
        print(f"  {name:14s} P@{EVAL_K} = {s:.2f}{mark}")

print("\n" + "=" * 40)
print("Aggregate P@3:")
for name, vals in totals.items():
    print(f"  {name:14s} {sum(vals) / len(vals):.2f}")
