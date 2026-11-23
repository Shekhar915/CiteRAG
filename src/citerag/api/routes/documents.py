import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from citerag.agent.pipeline import QAPipeline
from citerag.api.deps import get_pipeline
from citerag.core.config import get_settings
from citerag.schemas.documents import DocumentOut, UploadOut

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
def list_documents(pipeline: QAPipeline = Depends(get_pipeline)) -> list[DocumentOut]:
    return [DocumentOut(**vars(d)) for d in pipeline.list_documents()]


@router.post("", response_model=UploadOut)
async def upload_document(
    file: UploadFile, pipeline: QAPipeline = Depends(get_pipeline)
) -> UploadOut:
    """Index an uploaded PDF, Markdown or text file (re-uploads replace old chunks)."""
    name = Path(file.filename or "").name
    if not name:
        raise HTTPException(status_code=400, detail="file name is required")

    limit = get_settings().max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"file exceeds the {limit} byte limit")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / name
        path.write_bytes(data)
        try:
            if pipeline.is_up_to_date(path):
                return UploadOut(name=name, chunks_added=0, unchanged=True)
            return UploadOut(name=name, chunks_added=pipeline.ingest(path), unchanged=False)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/{name}", status_code=204)
def delete_document(name: str, pipeline: QAPipeline = Depends(get_pipeline)) -> None:
    if not pipeline.delete_document(name):
        raise HTTPException(status_code=404, detail="document not found")
