from typing import Dict, List

import pandas as pd


def _positive_feedback(row: pd.Series) -> str:
    improvement = float(row.get("improvement_rate", 0))
    if improvement > 5:
        return "Great progress. Keep following your current study plan."
    if improvement > 0:
        return "Progress is visible. Continue with regular revision and practice."
    return "Add weekly review sessions and monitor progress every test cycle."


def _weak_subject_plan(weak_subjects: List[str]) -> str:
    if not weak_subjects:
        return "No weak subjects detected. Focus on consistency and advanced practice."

    subject_text = ", ".join(weak_subjects)
    return (
        f"Marks below 40 in {subject_text}. Create a focused plan: 45 minutes daily practice, "
        "topic-wise mock tests, and teacher feedback twice a week."
    )


def _risk_alert(row: pd.Series) -> str:
    risk_level = row.get("risk_level", "Low")
    predicted_avg = float(row.get("predicted_average", 0))

    if risk_level == "High" or predicted_avg < 50:
        return "High risk alert: arrange mentoring, parent communication, and short-cycle assessments."
    if risk_level == "Medium":
        return "Medium risk: increase supervision and set measurable weekly targets."
    return "Low risk: maintain momentum with periodic enrichment tasks."


def build_recommendations(prediction_df: pd.DataFrame, subject_columns: List[str]) -> List[Dict]:
    recommendations: List[Dict] = []

    for _, row in prediction_df.iterrows():
        weak_subjects = [s for s in subject_columns if float(row.get(s, 0)) < 40]

        plan_parts = [
            _weak_subject_plan(weak_subjects),
            _positive_feedback(row),
            _risk_alert(row),
        ]

        recommendations.append(
            {
                "Name": row.get("Name", "Student"),
                "predicted_average": round(float(row.get("predicted_average", 0)), 2),
                "performer_level": row.get("performer_level", "Unknown"),
                "risk_level": row.get("risk_level", "Low"),
                "improvement_rate": round(float(row.get("improvement_rate", 0)), 2),
                "weak_subjects": weak_subjects,
                "recommendation": " ".join(plan_parts),
            }
        )

    return recommendations
