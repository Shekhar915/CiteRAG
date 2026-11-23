from pydantic import BaseModel


class DocumentOut(BaseModel):
    name: str
    chunks: int
    sha256: str


class UploadOut(BaseModel):
    name: str
    chunks_added: int
    unchanged: bool
