from functools import lru_cache

from fastapi import HTTPException

from citerag.agent.pipeline import QAPipeline, build_pipeline


@lru_cache
def _default_pipeline() -> QAPipeline:
    return build_pipeline()


def get_pipeline() -> QAPipeline:
    """FastAPI dependency; replaced with a fake pipeline in tests."""
    try:
        return _default_pipeline()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
