import pytest

from citerag.ingestion.chunker import chunk_text


def test_short_text_is_one_chunk():
    chunks = chunk_text("hello world", "doc.txt")
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].source == "doc.txt"
    assert chunks[0].index == 0


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, "doc.txt", size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # consecutive chunks share text because of the overlap
    assert chunks[1].start < chunks[0].end


def test_chunks_cover_the_whole_text():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, "doc.txt", size=150, overlap=30)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert "w499" in chunks[-1].text


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ", "doc.txt") == []


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        chunk_text("abc", "d", size=0)
    with pytest.raises(ValueError):
        chunk_text("abc", "d", size=10, overlap=10)
