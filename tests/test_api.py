import io
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.routes import prediction as prediction_routes
from app.services.pipeline_service import PipelineService


class DummyNLPService:
    def enrich(self, recommendations):
        enriched = []
        for item in recommendations:
            copy = dict(item)
            copy["performance_text"] = f"{item.get('Name', 'Student')} performance context"
            copy["nlp_label"] = "NEUTRAL"
            copy["nlp_score"] = 0.0
            copy["nlp_summary"] = "Neutral learning signal detected."
            enriched.append(copy)
        return enriched


def _csv_bytes() -> bytes:
    df = pd.DataFrame(
        {
            "Name": ["A", "B", "C", "D", "E", "F"],
            "Math": [45, 55, 65, 75, 35, 85],
            "Science": [50, 58, 66, 72, 40, 88],
            "English": [48, 54, 68, 74, 36, 82],
            "attendance": [62, 70, 76, 84, 55, 91],
            "previous_avg": [44, 52, 63, 70, 34, 80],
        }
    )
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def test_api_predict_requires_upload_when_no_csv(tmp_path, monkeypatch):
    service = PipelineService(
        upload_path=Path(tmp_path / "latest_upload.csv"),
        model_path=Path(tmp_path / "student_model.joblib"),
        nlp_service=DummyNLPService(),
    )

    app.dependency_overrides[prediction_routes.get_pipeline_service] = lambda: service

    client = TestClient(app)
    response = client.get("/predict")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "NO_UPLOADED_FILE"

    app.dependency_overrides.clear()


def test_full_api_flow_upload_predict_recommend(tmp_path, monkeypatch):
    service = PipelineService(
        upload_path=Path(tmp_path / "latest_upload.csv"),
        model_path=Path(tmp_path / "student_model.joblib"),
        nlp_service=DummyNLPService(),
    )

    app.dependency_overrides[prediction_routes.get_pipeline_service] = lambda: service

    client = TestClient(app)

    upload_response = client.post(
        "/upload",
        files={"file": ("students.csv", _csv_bytes(), "text/csv")},
    )
    assert upload_response.status_code == 200
    assert upload_response.json()["rows"] == 6

    predict_response = client.get("/predict")
    assert predict_response.status_code == 200
    predictions = predict_response.json()["predictions"]
    assert len(predictions) == 6
    assert "predicted_average" in predictions[0]

    recommend_response = client.get("/recommend")
    assert recommend_response.status_code == 200
    recommendations = recommend_response.json()["recommendations"]
    assert len(recommendations) == 6
    assert "recommendation" in recommendations[0]
    assert "nlp_summary" in recommendations[0]

    app.dependency_overrides.clear()
