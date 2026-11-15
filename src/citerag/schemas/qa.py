from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    chunk_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    snippet: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True