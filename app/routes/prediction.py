from fastapi import APIRouter, Depends, File, UploadFile

from app.models.schemas import (
    PredictResponse,
    RecommendResponse,
    UploadFileMeta,
    UploadResponse,
)
from app.services.container import get_pipeline_service
from app.services.pipeline_service import PipelineService
from app.utils.exceptions import AppException


router = APIRouter(prefix="", tags=["Prediction"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload CSV/Excel and train model",
)
async def upload_data(
    file: UploadFile = File(...),
    service: PipelineService = Depends(get_pipeline_service),
) -> UploadResponse:
    try:
        meta = UploadFileMeta(filename=file.filename)
    except Exception as exc:
        raise AppException(
            "Invalid file extension.",
            status_code=400,
            code="INVALID_FILE_TYPE",
            details={"reason": str(exc)},
        ) from exc

    content = await file.read()
    payload = service.upload_and_train(filename=meta.filename.lower(), content=content)
    return UploadResponse(**payload)


@router.get(
    "/predict",
    response_model=PredictResponse,
    summary="Get predictions for latest uploaded dataset",
)
def predict(
    service: PipelineService = Depends(get_pipeline_service),
) -> PredictResponse:
    payload = service.get_predictions()
    return PredictResponse(**payload)


@router.get(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get recommendations for latest predictions",
)
def recommend(
    service: PipelineService = Depends(get_pipeline_service),
) -> RecommendResponse:
    payload = service.get_recommendations()
    return RecommendResponse(**payload)
