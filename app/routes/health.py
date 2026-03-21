from fastapi import APIRouter

from app.models.schemas import HealthResponse


router = APIRouter(tags=["Health"])


@router.get("/", response_model=HealthResponse, summary="Health check")
def health_check() -> HealthResponse:
    return HealthResponse(message="Student Performance Predictor API is running.")
