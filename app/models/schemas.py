from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorDetail


class UploadFileMeta(BaseModel):
    filename: str = Field(min_length=1)

    @field_validator("filename")
    @classmethod
    def validate_extension(cls, value: str) -> str:
        lower_name = value.lower()
        if not lower_name.endswith((".csv", ".xlsx", ".xls")):
            raise ValueError("Only CSV and Excel files are supported.")
        return value


class HealthResponse(BaseModel):
    message: str


class UploadResponse(BaseModel):
    message: str
    rows: int
    columns: List[str]
    subject_columns: List[str]
    preview: List[Dict[str, Any]]
    target_mode: str


class PredictResponse(BaseModel):
    message: str
    total_students: int
    predictions: List[Dict[str, Any]]


class RecommendResponse(BaseModel):
    message: str
    recommendations: List[Dict[str, Any]]
