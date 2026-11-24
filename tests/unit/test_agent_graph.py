from types import SimpleNamespace

from citerag.agent.graph import run_agent
from citerag.ingestion.chunker import Chunk
from citerag.retrieval.vectorstore import SearchHit


class ScriptedLLM:
    """Replies by prompt type: grade, rewrite or answer."""

    def __init__(self, grades, rewrite="refund policy days", answer="It takes 14 days [1]."):
        self.grades = list(grades)
        self.rewrite = rewrite
        self.answer = answer
        self.calls = {"grade": 0, "rewrite": 0, "answer": 0}

    def invoke(self, messages):
        system = messages[0][1]
        if "judge whether" in system:
            self.calls["grade"] += 1
            return SimpleNamespace(content=self.grades.pop(0))
        if "Rewrite it" in system:
            self.calls["rewrite"] += 1
            return SimpleNamespace(content=self.rewrite)
        self.calls["answer"] += 1
        return SimpleNamespace(content=self.answer)


def hit(text="Refunds take 14 days."):
    return SearchHit(Chunk(text=text, source="p.txt", index=0, start=0, end=len(text)), 1.0)


def test_good_context_goes_straight_to_answer():
    llm = ScriptedLLM(grades=["yes"])
    response = run_agent("How long do refunds take?", lambda q: [hit()], llm)
    assert response.grounded is True
    assert llm.calls == {"grade": 1, "rewrite": 0, "answer": 1}


def test_weak_context_triggers_a_rewrite_then_succeeds():
    queries = []

    def search(query):
        queries.append(query)
        return [hit()]

    llm = ScriptedLLM(grades=["no", "yes"])
    response = run_agent("money back?", search, llm)

    assert queries == ["money back?", "refund policy days"]
    assert response.grounded is True
    assert llm.calls["rewrite"] == 1


def test_gives_up_after_max_attempts_without_answering():
    llm = ScriptedLLM(grades=["no", "no", "no"])
    response = run_agent("unanswerable", lambda q: [hit("unrelated")], llm, max_attempts=2)

    assert response.grounded is False
    assert response.citations == []
    assert llm.calls["answer"] == 0
    assert llm.calls["grade"] == 2


def test_no_hits_skips_grading_and_never_calls_answer():
    llm = ScriptedLLM(grades=[])
    response = run_agent("anything", lambda q: [], llm, max_attempts=2)

    assert response.grounded is False
    assert llm.calls["grade"] == 0
    assert llm.calls["answer"] == 0
