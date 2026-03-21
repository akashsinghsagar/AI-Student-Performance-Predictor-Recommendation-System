import pandas as pd

from recommendation import build_recommendations


def test_build_recommendations_flags_weak_subjects_and_risk():
    prediction_df = pd.DataFrame(
        [
            {
                "Name": "Alice",
                "Math": 35,
                "Science": 60,
                "English": 38,
                "predicted_average": 42,
                "performer_level": "Low",
                "risk_level": "High",
                "improvement_rate": -2,
            },
            {
                "Name": "Bob",
                "Math": 75,
                "Science": 78,
                "English": 80,
                "predicted_average": 77,
                "performer_level": "High",
                "risk_level": "Low",
                "improvement_rate": 8,
            },
        ]
    )

    output = build_recommendations(prediction_df, ["Math", "Science", "English"])

    assert len(output) == 2
    assert output[0]["Name"] == "Alice"
    assert "Math" in output[0]["weak_subjects"]
    assert "English" in output[0]["weak_subjects"]
    assert "High risk alert" in output[0]["recommendation"]

    assert output[1]["Name"] == "Bob"
    assert output[1]["weak_subjects"] == []
    assert "Great progress" in output[1]["recommendation"]
