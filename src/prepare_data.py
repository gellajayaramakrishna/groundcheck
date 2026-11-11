"""Convert RAGTruth (word-level hallucination spans) into sentence-level training pairs:
(premise = top-2 retrieved chunks of the source, hypothesis = one response sentence,
label = 1 if the sentence overlaps any human-marked hallucinated span, else 0)."""
import argparse
import json
import os
from collections import Counter, defaultdict

from tqdm import tqdm

from src.text_utils import chunk_text, overlaps, source_to_text, split_sentences


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_examples(responses, sources, retriever, top_k=2):
    by_source = defaultdict(list)
    for r in responses:
        by_source[r["source_id"]].append(r)

    examples = []
    for source_id, resps in tqdm(by_source.items(), desc="sources"):
        src = sources[source_id]
        chunks = chunk_text(source_to_text(src["source_info"]))
        if not chunks:
            continue
        chunk_embs = retriever.embed_passages(chunks)

        sent_rows = []
        for r in resps:
            spans = r["labels"]
            for s, e, sent in split_sentences(r["response"]):
                label = int(any(overlaps(s, e, sp["start"], sp["end"]) for sp in spans))
                sent_rows.append((r, sent, label))
        if not sent_rows:
            continue

        query_embs = retriever.embed_queries([row[1] for row in sent_rows])
        best, _ = retriever.top_k(query_embs, chunk_embs, k=top_k)
        for (r, sent, label), idx in zip(sent_rows, best):
            examples.append({
                "premise": "\n".join(chunks[i] for i in idx),
                "hypothesis": sent,
                "label": label,  # 1 = hallucinated, 0 = supported
                "task": src["task_type"],
                "source_id": source_id,
                "response_id": r["id"],
                "gen_model": r["model"],
                "split": r["split"],
            })
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ragtruth_dir", default="data/raw/RAGTruth/dataset")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_every", type=int, default=10, help="1 in N train sources goes to validation")
    args = ap.parse_args()

    from src.retriever import Retriever
    responses = load_jsonl(os.path.join(args.ragtruth_dir, "response.jsonl"))
    sources = {s["source_id"]: s for s in load_jsonl(os.path.join(args.ragtruth_dir, "source_info.jsonl"))}
    examples = build_examples(responses, sources, Retriever())

    # Validation is split BY SOURCE, so the same article never appears in both train and val.
    splits = {"train": [], "val": [], "test": []}
    for ex in examples:
        if ex["split"] == "test":
            splits["test"].append(ex)
        elif int(ex["source_id"]) % args.val_every == 0:
            splits["val"].append(ex)
        else:
            splits["train"].append(ex)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in splits.items():
        write_jsonl(rows, os.path.join(args.out_dir, f"{name}.jsonl"))
        c = Counter(r["label"] for r in rows)
        print(f"{name:5s}: {len(rows):6d} sentences | hallucinated {c[1]:5d} ({c[1] / max(len(rows), 1):.1%})")


if __name__ == "__main__":
    main()
