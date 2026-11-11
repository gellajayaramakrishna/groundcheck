from src.metrics import best_threshold, compute_metrics
from src.text_utils import chunk_text, overlaps, source_to_text, split_sentences


def test_split_keeps_offsets():
    text = "Revenue grew 12% in 2024. Dr. Rao said it was strong! Was it?"
    sents = split_sentences(text)
    assert [s for _, _, s in sents] == ["Revenue grew 12% in 2024.", "Dr. Rao said it was strong!", "Was it?"]
    for start, end, s in sents:
        assert text[start:end] == s


def test_decimal_numbers_not_split():
    assert len(split_sentences("The rate was 3.5 percent. It rose.")) == 2


def test_newlines_are_boundaries():
    assert len(split_sentences("- point one\n- point two")) == 2


def test_overlap():
    assert overlaps(0, 10, 5, 15)
    assert not overlaps(0, 10, 10, 20)


def test_chunking_covers_all_words():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), max_words=150, stride=120)
    assert chunks[-1].split()[-1] == "w399"
    assert all(len(c.split()) <= 150 for c in chunks)


def test_source_shapes():
    assert source_to_text("plain") == "plain"
    assert "Question: q" in source_to_text({"question": "q", "passages": "p"})
    text = source_to_text({"name": "Cafe", "review_info": [{"review_stars": 5, "review_text": "Great"}]})
    assert "name: Cafe" in text and "Great" in text


def test_metrics_threshold():
    y = [0, 0, 0, 1, 1]
    s = [0.1, 0.2, 0.3, 0.8, 0.9]
    t = best_threshold(y, s)
    assert compute_metrics(y, s, t)["f1"] == 1.0
