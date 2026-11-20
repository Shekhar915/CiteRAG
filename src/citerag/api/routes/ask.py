from fastapi import APIRouter, Depends

from citerag.agent.pipeline import QAPipeline
from citerag.api.deps import get_pipeline
from citerag.schemas.qa import AskRequest, AskResponse

router = APIRouter(prefix="/v1", tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, pipeline: QAPipeline = Depends(get_pipeline)) -> AskResponse:
    """Answer a question from the indexed documents, with citations."""
    return pipeline.ask(request.question, top_k=request.top_k)
