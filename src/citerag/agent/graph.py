from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from citerag.agent.answering import NO_ANSWER_TEXT, answer_question, build_context, content_text
from citerag.retrieval.vectorstore import SearchHit
from citerag.schemas.qa import AskResponse

SearchFn = Callable[[str], list[SearchHit]]

GRADE_PROMPT = (
    "You judge whether retrieved passages contain the information needed to answer "
    "a question. Reply with exactly one word: yes or no."
)
REWRITE_PROMPT = (
    "The search for this question returned weak results. Rewrite it as a short, "
    "specific search query using likely keywords from the source documents. "
    "Reply with only the rewritten query."
)


class AgentState(TypedDict, total=False):
    question: str
    query: str
    hits: list[SearchHit]
    attempts: int
    relevant: bool
    response: AskResponse


def build_agent_graph(search: SearchFn, llm: Any, max_attempts: int = 2):
    """Retrieve, grade the context, and rewrite the query when it is weak.

    The loop is bounded by `max_attempts` retrievals. If the context is still
    weak after that, the agent says so instead of guessing.
    """

    def retrieve(state: AgentState) -> dict:
        return {
            "hits": search(state["query"]),
            "attempts": state.get("attempts", 0) + 1,
        }

    def grade(state: AgentState) -> dict:
        hits = state.get("hits", [])
        if not hits:
            return {"relevant": False}
        user = f"Question: {state['question']}\n\nPassages:\n{build_context(hits)}"
        verdict = content_text(llm.invoke([("system", GRADE_PROMPT), ("user", user)]).content)
        return {"relevant": verdict.lower().startswith("yes")}

    def rewrite(state: AgentState) -> dict:
        user = f"Question: {state['question']}\nPrevious query: {state['query']}"
        text = content_text(llm.invoke([("system", REWRITE_PROMPT), ("user", user)]).content)
        return {"query": text or state["query"]}

    def answer(state: AgentState) -> dict:
        return {"response": answer_question(state["question"], state.get("hits", []), llm)}

    def give_up(state: AgentState) -> dict:
        return {"response": AskResponse(answer=NO_ANSWER_TEXT, citations=[], grounded=False)}

    def after_grade(state: AgentState) -> str:
        if state.get("relevant"):
            return "answer"
        return "rewrite" if state.get("attempts", 0) < max_attempts else "give_up"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("answer", answer)
    graph.add_node("give_up", give_up)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", after_grade, ["answer", "rewrite", "give_up"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("answer", END)
    graph.add_edge("give_up", END)
    return graph.compile()


def run_agent(question: str, search: SearchFn, llm: Any, max_attempts: int = 2) -> AskResponse:
    graph = build_agent_graph(search, llm, max_attempts)
    final = graph.invoke({"question": question, "query": question, "attempts": 0})
    return final["response"]
