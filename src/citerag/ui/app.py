import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run src/citerag/ui/app.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from citerag.agent.pipeline import build_pipeline  # noqa: E402
from citerag.core.logging import setup_logging  # noqa: E402
from citerag.ui.helpers import format_citation, ingest_uploads  # noqa: E402

setup_logging()
st.set_page_config(page_title="CiteRAG", page_icon="📚", layout="wide")


@st.cache_resource
def get_pipeline():
    return build_pipeline()


def render_answer(item: dict) -> None:
    if not item["grounded"]:
        st.warning(item["answer"])
        return
    st.markdown(item["answer"])
    for number, citation in enumerate(item["citations"], start=1):
        with st.expander(f"Source {number}: {format_citation(citation)}"):
            st.write(citation.snippet)


st.title("📚 CiteRAG")
st.caption("Upload documents, then ask questions. Answers come with citations.")

try:
    pipeline = get_pipeline()
except ValueError as exc:
    st.error(f"{exc}. Add it to your .env file and restart the app.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("1. Documents")
    files = st.file_uploader(
        "PDF, Markdown or text files", type=["pdf", "md", "markdown", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", disabled=not files, use_container_width=True):
        with st.spinner("Indexing..."):
            try:
                for result in ingest_uploads(pipeline, [(f.name, f.getvalue()) for f in files]):
                    if result.error:
                        st.error(f"{result.name}: {result.error}")
                    elif result.unchanged:
                        st.info(f"{result.name}: already indexed, no changes.")
                    else:
                        st.success(f"{result.name}: indexed ({result.chunks} chunks).")
            except Exception as exc:  # network or API errors
                st.error(f"Indexing failed: {exc}")

    documents = pipeline.list_documents()
    if documents:
        st.subheader("Indexed")
        for doc in documents:
            name_col, delete_col = st.columns([4, 1])
            name_col.write(f"{doc.name} ({doc.chunks} chunks)")
            if delete_col.button("🗑", key=f"delete-{doc.name}", help=f"Remove {doc.name}"):
                pipeline.delete_document(doc.name)
                st.rerun()

    top_k = st.slider("Passages to retrieve", 1, 10, 5)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []

st.header("2. Ask")
for item in st.session_state.history:
    with st.chat_message("user"):
        st.write(item["question"])
    with st.chat_message("assistant"):
        render_answer(item)

question = st.chat_input("Ask a question about your documents")
if question:
    if not pipeline.list_documents():
        st.warning("Index at least one document first (sidebar).")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching and answering..."):
                try:
                    response = pipeline.ask(question, top_k=top_k)
                except Exception as exc:  # network or API errors
                    st.error(f"Could not answer: {exc}")
                    st.stop()
            item = {
                "question": question,
                "answer": response.answer,
                "grounded": response.grounded,
                "citations": response.citations,
            }
            st.session_state.history.append(item)
            render_answer(item)
