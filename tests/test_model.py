import pandas as pd

from model import StudentPerformanceModel


def _sample_training_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [f"Student_{i}" for i in range(1, 9)],
            "Math": [35, 45, 50, 60, 70, 80, 90, 55],
            "Science": [40, 50, 55, 65, 75, 85, 92, 58],
            "English": [30, 48, 52, 62, 72, 82, 88, 54],
            "attendance": [58, 65, 72, 78, 85, 90, 95, 70],
            "previous_avg": [38, 44, 49, 59, 69, 79, 87, 52],
        }
    )


def test_train_and_predict_generates_expected_columns(tmp_path):
    model_path = tmp_path / "student_model.joblib"
    model = StudentPerformanceModel(model_path=str(model_path))

    train_df = _sample_training_df()
    model.train(train_df)

    predictions = model.predict(train_df)

    assert not predictions.empty
    assert "predicted_average" in predictions.columns
    assert "performer_level" in predictions.columns
    assert "risk_level" in predictions.columns
    assert "weak_subjects" in predictions.columns
    assert "predicted_Math" in predictions.columns
    assert set(predictions["performer_level"].unique()).issubset({"High", "Medium", "Low"})
    assert set(predictions["risk_level"].unique()).issubset({"High", "Medium", "Low"})
    assert model_path.exists()


def test_preprocess_raises_when_no_subject_columns(tmp_path):
    model = StudentPerformanceModel(model_path=str(tmp_path / "student_model.joblib"))
    bad_df = pd.DataFrame({"Name": ["A", "B"], "attendance": [80, 90]})

    try:
        model.preprocess(bad_df, fit=True)
        assert False, "Expected ValueError for missing subject columns"
    except ValueError as exc:
        assert "No subject/marks columns" in str(exc)
