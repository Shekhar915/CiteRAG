import re
from typing import Any

from citerag.retrieval.vectorstore import SearchHit
from citerag.schemas.qa import AskResponse, Citation

NOT_FOUND_TOKEN = "NOT_FOUND"
NO_ANSWER_TEXT = "I couldn't find the answer to this in the provided documents."
SNIPPET_CHARS = 200

SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages provided. "
    "Cite the passages you use with their numbers in square brackets, like [1]. "
    f"If the context does not contain the answer, reply with exactly {NOT_FOUND_TOKEN}. "
    "Never use outside knowledge."
)

_CITATION_MARK = re.compile(r"\[(\d+)\]")


def build_context(hits: list[SearchHit]) -> str:
    lines = []
    for number, hit in enumerate(hits, start=1):
        page = f", page {hit.chunk.page}" if hit.chunk.page else ""
        lines.append(f"[{number}] ({hit.chunk.source}{page})\n{hit.chunk.text}")
    return "\n\n".join(lines)


def build_messages(question: str, hits: list[SearchHit]) -> list[tuple[str, str]]:
    user = f"Context:\n{build_context(hits)}\n\nQuestion: {question}"
    return [("system", SYSTEM_PROMPT), ("user", user)]


def to_citation(hit: SearchHit) -> Citation:
    return Citation(
        source=hit.chunk.source,
        chunk_index=hit.chunk.index,
        page=hit.chunk.page,
        snippet=hit.chunk.text[:SNIPPET_CHARS],
    )


def content_text(content: Any) -> str:
    """Normalise a model response, which may be a string or a list of parts."""
    if isinstance(content, list):
        content = "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return str(content or "").strip()


def cited_hits(answer: str, hits: list[SearchHit]) -> list[SearchHit]:
    """Return the hits the answer cites; fall back to all hits if it cites none."""
    numbers = {int(n) for n in _CITATION_MARK.findall(answer)}
    chosen = [hit for number, hit in enumerate(hits, start=1) if number in numbers]
    return chosen or hits


def answer_question(question: str, hits: list[SearchHit], llm: Any) -> AskResponse:
    """Answer from the retrieved hits, or say so when the documents don't cover it."""
    if not hits:
        return AskResponse(answer=NO_ANSWER_TEXT, citations=[], grounded=False)

    text = content_text(llm.invoke(build_messages(question, hits)).content)
    if not text or text.strip(" .") == NOT_FOUND_TOKEN:
        return AskResponse(answer=NO_ANSWER_TEXT, citations=[], grounded=False)

    return AskResponse(
        answer=text,
        citations=[to_citation(h) for h in cited_hits(text, hits)],
        grounded=True,
    )
