import tempfile
from dataclasses import dataclass
from pathlib import Path

from citerag.agent.pipeline import QAPipeline
from citerag.schemas.qa import Citation


@dataclass(frozen=True)
class UploadResult:
    name: str
    chunks: int = 0
    error: str | None = None
    unchanged: bool = False


def ingest_uploads(pipeline: QAPipeline, uploads: list[tuple[str, bytes]]) -> list[UploadResult]:
    """Save uploaded files to a temp folder and index them, one result per file.

    A bad file (wrong type, too large) is reported without stopping the others.
    """
    results: list[UploadResult] = []
    with tempfile.TemporaryDirectory() as folder:
        for name, data in uploads:
            safe_name = Path(name).name
            path = Path(folder) / safe_name
            path.write_bytes(data)
            try:
                if pipeline.is_up_to_date(path):
                    results.append(UploadResult(safe_name, unchanged=True))
                    continue
                results.append(UploadResult(safe_name, chunks=pipeline.ingest(path)))
            except ValueError as exc:
                results.append(UploadResult(safe_name, error=str(exc)))
    return results


def format_citation(citation: Citation) -> str:
    parts = [citation.source]
    if citation.page:
        parts.append(f"page {citation.page}")
    parts.append(f"chunk {citation.chunk_index}")
    return " · ".join(parts)
