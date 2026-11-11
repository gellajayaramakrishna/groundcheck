"""Fine-tune DeBERTa-v3-base (already trained on general NLI) to classify each
(evidence, sentence) pair as supported (0) or hallucinated (1). Run this on a GPU (Kaggle/Colab)."""
import argparse
import os

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from src.metrics import compute_metrics
from src.prepare_data import load_jsonl


class WeightedTrainer(Trainer):
    """Hallucinated sentences are only ~11% of the data, so their mistakes get a bigger weight in the loss."""

    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits.float(), labels)
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--base_model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--out_dir", default="models/groundcheck")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--train_limit", type=int, default=None, help="for a quick smoke test")
    ap.add_argument("--cpu", action="store_true", help="force CPU (avoids running out of Mac GPU memory)")
    args = ap.parse_args()
    set_seed(42)

    train_rows = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))[:args.train_limit]
    val_rows = load_jsonl(os.path.join(args.data_dir, "val.jsonl"))
    if args.train_limit:
        val_rows = val_rows[:max(args.train_limit // 5, 200)]

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def to_ds(rows):
        ds = Dataset.from_dict({"premise": [r["premise"] for r in rows],
                                "hypothesis": [r["hypothesis"] for r in rows],
                                "labels": [r["label"] for r in rows]})
        return ds.map(lambda b: tok(b["premise"], b["hypothesis"], truncation="only_first",
                                    max_length=args.max_length),
                      batched=True, remove_columns=["premise", "hypothesis"])

    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)

    counts = np.bincount([r["label"] for r in train_rows], minlength=2)
    weights = torch.tensor(len(train_rows) / (2.0 * counts), dtype=torch.float)
    print("label counts:", counts.tolist(), "class weights:", weights.tolist())

    # Reuse the NLI-trained body, replace the 3-way head with a fresh 2-way head.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, ignore_mismatched_sizes=True,
        id2label={0: "supported", 1: "hallucinated"}, label2id={"supported": 0, "hallucinated": 1})

    def metrics_fn(pred):
        logits, labels = pred
        probs = torch.softmax(torch.tensor(logits).float(), dim=-1)[:, 1].numpy()
        return compute_metrics(labels, probs, 0.5)

    targs = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_ratio=0.06,
        weight_decay=0.01,
        eval_strategy="steps", eval_steps=1000,
        save_strategy="steps", save_steps=1000, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="pr_auc", greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        group_by_length=True,
        report_to="none",
        use_cpu=args.cpu,
    )
    trainer = WeightedTrainer(class_weights=weights, model=model, args=targs,
                              train_dataset=train_ds, eval_dataset=val_ds,
                              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics_fn)
    trainer.train()
    trainer.save_model(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("final validation:", trainer.evaluate())
    print("saved model to", args.out_dir)


if __name__ == "__main__":
    main()
