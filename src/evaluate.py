"""Evaluate the fine-tuned model: pick the decision threshold on validation, then report
in-domain RAGTruth test metrics (sentence, per-task, response level) and out-of-domain HaluBench."""
import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

from src.metrics import best_threshold, compute_metrics
from src.prepare_data import load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/groundcheck")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--halubench_n", type=int, default=2000, help="random HaluBench examples (0 to skip)")
    args = ap.parse_args()

    from src.checker import GroundChecker, NLIScorer
    scorer = NLIScorer(args.model_dir)
    val = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    test = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))

    sv = scorer.score([r["premise"] for r in val], [r["hypothesis"] for r in val])
    thr = best_threshold([r["label"] for r in val], sv)
    json.dump({"threshold": thr}, open(os.path.join(args.model_dir, "threshold.json"), "w"))
    print(f"threshold chosen on validation: {thr:.2f}")

    st = scorer.score([r["premise"] for r in test], [r["hypothesis"] for r in test])
    res = {"ragtruth_sentence": compute_metrics([r["label"] for r in test], st, thr)}

    by_task = defaultdict(list)
    for r, s in zip(test, st):
        by_task[r["task"]].append((r["label"], s))
    res["ragtruth_by_task"] = {t: compute_metrics([a for a, _ in v], [b for _, b in v], thr)
                               for t, v in by_task.items()}

    # Response level: a response is hallucinated if ANY of its sentences is.
    resp = defaultdict(lambda: [0, 0.0])
    for r, s in zip(test, st):
        resp[r["response_id"]][0] = max(resp[r["response_id"]][0], r["label"])
        resp[r["response_id"]][1] = max(resp[r["response_id"]][1], float(s))
    res["ragtruth_response"] = compute_metrics([v[0] for v in resp.values()],
                                               [v[1] for v in resp.values()], thr)

    if args.halubench_n:
        from datasets import load_dataset
        hb = list(load_dataset("PatronusAI/HaluBench", split="test"))
        random.Random(42).shuffle(hb)
        hb = hb[:args.halubench_n]
        checker = GroundChecker(args.model_dir, threshold=thr)
        checker.scorer = scorer
        labels, scores, srcs = [], [], []
        for ex in hb:
            out = checker.check(ex["passage"], ex["answer"], question=ex["question"])
            if not out:
                continue
            labels.append(int(ex["label"] == "FAIL"))
            scores.append(max(o["hallucination_prob"] for o in out))
            srcs.append(ex["source_ds"])
        res["halubench"] = compute_metrics(labels, scores, thr)
        groups = defaultdict(list)
        for l, s, d in zip(labels, scores, srcs):
            groups[d].append((l, s))
        res["halubench_by_source"] = {d: compute_metrics([a for a, _ in v], [b for _, b in v], thr)
                                      for d, v in groups.items()}

    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/eval.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
