from types import SimpleNamespace

from citerag.agent.answering import (
    NO_ANSWER_TEXT,
    answer_question,
    build_context,
    build_messages,
    cited_hits,
    content_text,
)
from citerag.ingestion.chunker import Chunk
from citerag.retrieval.vectorstore import SearchHit


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.reply)


def hit(text, index=0, source="a.pdf", page=None):
    chunk = Chunk(text=text, source=source, index=index, start=0, end=len(text), page=page)
    return SearchHit(chunk, 1.0)


def test_context_numbers_passages_and_shows_pages():
    context = build_context([hit("first", 0, page=3), hit("second", 1, source="b.md")])
    assert "[1] (a.pdf, page 3)\nfirst" in context
    assert "[2] (b.md)\nsecond" in context


def test_messages_contain_question_and_context():
    system, user = build_messages("What is X?", [hit("X is a thing")])
    assert system[0] == "system"
    assert "What is X?" in user[1]
    assert "X is a thing" in user[1]


def test_no_hits_skips_the_llm():
    llm = FakeLLM("should not be used")
    response = answer_question("q", [], llm)
    assert response.grounded is False
    assert response.answer == NO_ANSWER_TEXT
    assert llm.calls == 0


def test_not_found_token_becomes_ungrounded_answer():
    response = answer_question("q", [hit("unrelated")], FakeLLM("NOT_FOUND"))
    assert response.grounded is False
    assert response.citations == []


def test_answer_returns_only_cited_passages():
    hits = [hit("alpha", 0), hit("beta", 1), hit("gamma", 2)]
    response = answer_question("q", hits, FakeLLM("Beta is the answer [2]."))
    assert response.grounded is True
    assert [c.chunk_index for c in response.citations] == [1]


def test_answer_without_markers_cites_all_hits():
    hits = [hit("alpha", 0), hit("beta", 1)]
    assert len(cited_hits("no markers here", hits)) == 2


def test_out_of_range_marker_is_ignored():
    hits = [hit("alpha", 0)]
    assert cited_hits("see [9]", hits) == hits


def test_content_text_handles_part_lists():
    assert content_text([{"text": "hel"}, "lo"]) == "hello"
    assert content_text(None) == ""
