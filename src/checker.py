"""Step 2 of GroundCheck: an NLI model scores how likely each sentence is NOT supported
by its evidence. Works with our fine-tuned model (2 labels) or an off-the-shelf MNLI model (3 labels)."""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.retriever import Retriever, pick_device
from src.text_utils import chunk_text, split_sentences

DEFAULT_ZERO_SHOT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLIScorer:
    def __init__(self, model_path=DEFAULT_ZERO_SHOT, device=None, max_length=512):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device).eval()
        self.max_length = max_length
        labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        if "hallucinated" in labels.values():
            self.mode, self.idx = "halluc", [i for i, l in labels.items() if l == "hallucinated"][0]
        else:  # MNLI-style model: hallucination score = 1 - P(entailment)
            self.mode, self.idx = "nli", [i for i, l in labels.items() if l.startswith("entail")][0]

    @torch.no_grad()
    def score(self, premises, hypotheses, batch_size=16):
        out = []
        for i in range(0, len(premises), batch_size):
            enc = self.tokenizer(premises[i:i + batch_size], hypotheses[i:i + batch_size],
                                 truncation="only_first", max_length=self.max_length,
                                 padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)[:, self.idx].cpu().numpy()
            out.append(probs if self.mode == "halluc" else 1.0 - probs)
        return np.concatenate(out) if out else np.array([])


class GroundChecker:
    """Full pipeline: split answer into sentences -> retrieve evidence -> score each sentence."""

    def __init__(self, model_path=None, threshold=None):
        model_path = model_path or (DEFAULT_ZERO_SHOT if not os.path.isdir("models/groundcheck")
                                    else "models/groundcheck")
        self.model_path = model_path
        self.retriever = Retriever()
        self.scorer = NLIScorer(model_path)
        thr_file = os.path.join(model_path, "threshold.json")
        if threshold is None and os.path.exists(thr_file):
            threshold = json.load(open(thr_file))["threshold"]
        self.threshold = 0.5 if threshold is None else threshold

    def check(self, document, answer, question=None, top_k=2):
        chunks = chunk_text(document)
        sentences = [s for _, _, s in split_sentences(answer)]
        if not chunks or not sentences:
            return []
        # Very short single answers ("Yes", "$40") only make sense together with the question.
        hyps = sentences
        if question and len(sentences) == 1 and len(sentences[0].split()) < 8:
            hyps = [f"Question: {question} Answer: {sentences[0]}"]
        chunk_embs = self.retriever.embed_passages(chunks)
        best, _ = self.retriever.top_k(self.retriever.embed_queries(hyps), chunk_embs, k=top_k)
        premises = ["\n".join(chunks[i] for i in idx) for idx in best]
        scores = self.scorer.score(premises, hyps)
        return [{"sentence": s, "hallucination_prob": round(float(p), 4),
                 "supported": bool(p < self.threshold), "evidence": prem}
                for s, p, prem in zip(sentences, scores, premises)]
