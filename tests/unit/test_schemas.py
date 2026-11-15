import pytest
from pydantic import ValidationError

from citerag.schemas.qa import AskRequest, AskResponse, Citation


def test_ask_request_defaults():
    assert AskRequest(question="What is RAG?").top_k == 5


def test_ask_request_rejects_short_question_and_bad_top_k():
    with pytest.raises(ValidationError):
        AskRequest(question="hi")
    with pytest.raises(ValidationError):
        AskRequest(question="valid question", top_k=0)


def test_ask_response_holds_citations():
    response = AskResponse(answer="x", citations=[Citation(source="a.pdf", chunk_index=2, page=3)])
    assert response.citations[0].source == "a.pdf"
    assert response.grounded is True