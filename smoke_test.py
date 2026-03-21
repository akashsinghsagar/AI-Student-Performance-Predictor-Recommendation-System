import argparse
import io
import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import main
from model import StudentPerformanceModel


class DummyNLPService:
    def enrich(self, recommendations):
        enriched = []
        for item in recommendations:
            rec = dict(item)
            rec["performance_text"] = f"{rec.get('Name', 'Student')} performance context"
            rec["nlp_label"] = "NEUTRAL"
            rec["nlp_score"] = 0.0
            rec["nlp_summary"] = "Neutral learning signal detected."
            enriched.append(rec)
        return enriched


def prepare_dataframe(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    if "Name" not in df.columns:
        if "first_name" in df.columns and "last_name" in df.columns:
            df["Name"] = (df["first_name"].fillna("") + " " + df["last_name"].fillna(" ")).str.strip()
        elif "first_name" in df.columns:
            df["Name"] = df["first_name"].fillna("Student")
        else:
            df["Name"] = [f"Student_{i + 1}" for i in range(len(df))]

    subject_cols = [c for c in df.columns if c.endswith("_score")]
    if not subject_cols:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        ignored = {"id", "absence_days", "weekly_self_study_hours"}
        subject_cols = [c for c in numeric_cols if c not in ignored]

    if not subject_cols:
        raise ValueError("No subject-like numeric columns found for smoke test.")

    if "attendance" not in df.columns:
        if "absence_days" in df.columns:
            df["attendance"] = (100 - (df["absence_days"].fillna(0) * 2)).clip(0, 100)
        else:
            df["attendance"] = 75

    if "previous_avg" not in df.columns:
        avg = df[subject_cols].mean(axis=1)
        df["previous_avg"] = (avg - 5).clip(0, 100)

    model_df = df[["Name"] + subject_cols + ["attendance", "previous_avg"]].copy()
    return model_df


def run_smoke_test(input_csv: Path, sample_rows: int) -> int:
    model_df = prepare_dataframe(input_csv)

    main.state.latest_df = pd.DataFrame()
    main.state.latest_predictions = pd.DataFrame()
    main.state.latest_recommendations = []
    main.model_service = StudentPerformanceModel(model_path="artifacts/smoke_student_model.joblib")
    main.nlp_service = DummyNLPService()

    client = TestClient(main.app)

    csv_buffer = io.StringIO()
    model_df.to_csv(csv_buffer, index=False)
    payload = csv_buffer.getvalue().encode("utf-8")

    upload = client.post("/upload", files={"file": ("smoke_input.csv", payload, "text/csv")})
    if upload.status_code != 200:
        print("Upload failed:")
        print(upload.text)
        return 1

    predict = client.get("/predict")
    if predict.status_code != 200:
        print("Predict failed:")
        print(predict.text)
        return 1

    recommend = client.get("/recommend")
    if recommend.status_code != 200:
        print("Recommend failed:")
        print(recommend.text)
        return 1

    pred_records = predict.json().get("predictions", [])
    rec_records = recommend.json().get("recommendations", [])

    print("Smoke test passed.")
    print(f"Input rows: {len(model_df)}")
    print(f"Predictions returned: {len(pred_records)}")
    print(f"Recommendations returned: {len(rec_records)}")

    print("\nSample predictions:")
    print(json.dumps(pred_records[:sample_rows], indent=2))

    print("\nSample recommendations:")
    print(json.dumps(rec_records[:sample_rows], indent=2))

    return 0


def main_cli() -> int:
    parser = argparse.ArgumentParser(description="One-command smoke test for student performance API endpoints.")
    parser.add_argument("--csv", default="student-scores.csv", help="Path to input CSV file")
    parser.add_argument("--sample", type=int, default=2, help="Number of sample records to print")
    args = parser.parse_args()

    input_csv = Path(args.csv)
    if not input_csv.exists():
        print(f"CSV not found: {input_csv}")
        return 1

    return run_smoke_test(input_csv=input_csv, sample_rows=max(1, args.sample))


if __name__ == "__main__":
    raise SystemExit(main_cli())
