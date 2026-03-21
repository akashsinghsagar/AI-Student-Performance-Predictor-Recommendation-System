import io
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List

import pandas as pd

from app.services.model_service import StudentPerformanceModel
from app.services.nlp_service import PerformanceNLPAnalyzer
from app.services.recommendation_service import build_recommendations
from app.utils.exceptions import AppException
from app.utils.logger import get_logger


logger = get_logger(__name__)


class PipelineService:
    def __init__(
        self,
        upload_path: Path,
        model_path: Path,
        nlp_service: PerformanceNLPAnalyzer | None = None,
    ) -> None:
        self._lock = RLock()
        self.upload_path = upload_path
        self.model_service = StudentPerformanceModel(model_path=model_path)
        self.nlp_service = nlp_service or PerformanceNLPAnalyzer()

        self.latest_df = pd.DataFrame()
        self.latest_predictions = pd.DataFrame()
        self.latest_recommendations: List[Dict[str, Any]] = []

        self.upload_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def parse_uploaded_content(filename: str, content: bytes) -> pd.DataFrame:
        if filename.endswith(".csv"):
            text = content.decode("utf-8", errors="ignore")
            return pd.read_csv(io.StringIO(text))

        return pd.read_excel(io.BytesIO(content))

    def upload_and_train(self, filename: str, content: bytes) -> Dict[str, Any]:
        with self._lock:
            try:
                df = self.parse_uploaded_content(filename=filename, content=content)
            except Exception as exc:
                logger.warning("Failed to parse uploaded file: %s", exc)
                raise AppException(
                    "Could not parse uploaded file.",
                    status_code=400,
                    code="INVALID_FILE_CONTENT",
                    details={"reason": str(exc)},
                ) from exc

            if df.empty:
                raise AppException(
                    "Uploaded file is empty.",
                    status_code=400,
                    code="EMPTY_DATASET",
                )

            self.latest_df = df
            self.latest_df.to_csv(self.upload_path, index=False)

            try:
                self.model_service.train(self.latest_df)
                self.latest_predictions = self.model_service.predict(self.latest_df)
            except Exception as exc:
                logger.exception("Training failed.")
                raise AppException(
                    "Model training failed.",
                    status_code=500,
                    code="TRAINING_FAILED",
                    details={"reason": str(exc)},
                ) from exc

            return {
                "message": "File uploaded and model trained successfully.",
                "rows": len(df),
                "columns": df.columns.tolist(),
                "subject_columns": self.model_service.subject_columns,
                "preview": df.head(10).to_dict(orient="records"),
                "target_mode": self.model_service.target_mode,
            }

    def get_predictions(self) -> Dict[str, Any]:
        with self._lock:
            if self.latest_df.empty:
                if self.upload_path.exists():
                    self.latest_df = pd.read_csv(self.upload_path)
                else:
                    raise AppException(
                        "No uploaded file found.",
                        status_code=400,
                        code="NO_UPLOADED_FILE",
                    )

            try:
                self.latest_predictions = self.model_service.predict(self.latest_df)
            except Exception as exc:
                logger.exception("Prediction failed.")
                raise AppException(
                    "Prediction failed.",
                    status_code=500,
                    code="PREDICTION_FAILED",
                    details={"reason": str(exc)},
                ) from exc

            records = self.latest_predictions.to_dict(orient="records")
            return {
                "message": "Predictions generated successfully.",
                "total_students": len(records),
                "predictions": records,
            }

    def get_recommendations(self) -> Dict[str, Any]:
        with self._lock:
            if self.latest_predictions.empty:
                _ = self.get_predictions()

            try:
                recommendations = build_recommendations(
                    prediction_df=self.latest_predictions,
                    subject_columns=self.model_service.subject_columns,
                )
                recommendations = self.nlp_service.enrich(recommendations)
                self.latest_recommendations = recommendations
            except Exception as exc:
                logger.exception("Recommendation generation failed.")
                raise AppException(
                    "Recommendation generation failed.",
                    status_code=500,
                    code="RECOMMENDATION_FAILED",
                    details={"reason": str(exc)},
                ) from exc

            return {
                "message": "Recommendations generated successfully.",
                "recommendations": self.latest_recommendations,
            }
