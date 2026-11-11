"""Two simple baselines on the RAGTruth test set, to know what 'good' looks like:
(1) embedding similarity between sentence and evidence, (2) an off-the-shelf NLI model (no training)."""
import argparse
import json
import os

import numpy as np

from src.metrics import best_threshold, compute_metrics
from src.prepare_data import load_jsonl


def similarity_scores(rows, retriever):
    q = retriever.embed_queries([r["hypothesis"] for r in rows])
    p = retriever.embed_passages([r["premise"] for r in rows])
    return 1.0 - np.sum(q * p, axis=1)  # low similarity -> high hallucination score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--limit", type=int, default=None, help="use first N rows per split (faster on a laptop)")
    args = ap.parse_args()

    from src.checker import NLIScorer
    from src.retriever import Retriever
    val = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))[:args.limit]
    test = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))[:args.limit]
    yv, yt = [r["label"] for r in val], [r["label"] for r in test]
    results = {}

    ret = Retriever()
    sv, st = similarity_scores(val, ret), similarity_scores(test, ret)
    results["similarity_bge_small"] = compute_metrics(yt, st, best_threshold(yv, sv))
    print("similarity:", results["similarity_bge_small"])

    nli = NLIScorer()
    sv = nli.score([r["premise"] for r in val], [r["hypothesis"] for r in val])
    st = nli.score([r["premise"] for r in test], [r["hypothesis"] for r in test])
    results["zero_shot_nli_deberta_v3_base"] = compute_metrics(yt, st, best_threshold(yv, sv))
    print("zero-shot NLI:", results["zero_shot_nli_deberta_v3_base"])

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/baselines.json", "w"), indent=2)


if __name__ == "__main__":
    main()
