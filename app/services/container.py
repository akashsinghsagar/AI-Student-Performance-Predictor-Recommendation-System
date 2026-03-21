from functools import lru_cache

from app.services.pipeline_service import PipelineService
from app.utils.config import get_settings


@lru_cache(maxsize=1)
def get_pipeline_service() -> PipelineService:
    settings = get_settings()
    return PipelineService(upload_path=settings.upload_path, model_path=settings.model_path)
