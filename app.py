import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st


def _normalize_api_url(raw_url: str) -> str:
    cleaned = (raw_url or "").strip()
    cleaned = cleaned.rstrip("/")
    if not cleaned:
        cleaned = "http://127.0.0.1:8000"
    return cleaned


API_URL = _normalize_api_url(os.getenv("API_URL", "http://127.0.0.1:8000"))


def _api_get(path: str, timeout: int = 120) -> Dict:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def _api_upload(file_obj) -> Dict:
    filename = file_obj.name.lower()
    content_type = "application/octet-stream"
    if filename.endswith(".csv"):
        content_type = "text/csv"
    elif filename.endswith(".xlsx"):
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif filename.endswith(".xls"):
        content_type = "application/vnd.ms-excel"

    files = {"file": (file_obj.name, file_obj.getvalue(), content_type)}
    response = requests.post(f"{API_URL}/upload", files=files, timeout=180)
    response.raise_for_status()
    return response.json()


def _init_state() -> None:
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []
    if "recommendations" not in st.session_state:
        st.session_state["recommendations"] = []
    if "upload_preview" not in st.session_state:
        st.session_state["upload_preview"] = []
    if "subject_columns" not in st.session_state:
        st.session_state["subject_columns"] = []


def _subject_columns_from_predictions(pred_df: pd.DataFrame) -> List[str]:
    excluded = {
        "Name",
        "attendance",
        "avg_mark",
        "improvement_rate",
        "predicted_average",
        "performer_level",
        "risk_level",
        "weak_subjects",
    }
    return [
        col
        for col in pred_df.columns
        if col not in excluded and not col.startswith("predicted_")
    ]


def _predicted_columns(pred_df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in pred_df.columns
        if col.startswith("predicted_") and col != "predicted_average"
    ]


def _plot_subject_bar(pred_df: pd.DataFrame, subjects: List[str]) -> None:
    if pred_df.empty or not subjects:
        st.info("Subject-wise bar chart will appear once predictions are available.")
        return

    means = pred_df[subjects].mean(numeric_only=True).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(means.index, means.values, color="#38bdf8")
    ax.set_title("Subject-Wise Average Marks")
    ax.set_ylabel("Average Marks")
    ax.set_xlabel("Subject")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    st.pyplot(fig)


def _plot_performance_line(pred_df: pd.DataFrame, student_name: str, subjects: List[str]) -> None:
    if pred_df.empty or not student_name or not subjects:
        st.info("Performance trend chart will appear after selecting a student.")
        return

    student_row = pred_df[pred_df["Name"] == student_name]
    if student_row.empty:
        st.warning("Selected student data not found.")
        return

    row = student_row.iloc[0]
    pred_cols_map = {f"predicted_{s}": s for s in subjects}

    actual_y = [float(row.get(s, 0.0)) for s in subjects]
    predicted_y = [float(row.get(f"predicted_{s}", 0.0)) for s in subjects]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(subjects, actual_y, marker="o", linewidth=2.5, color="#0ea5e9", label="Current")
    if any(f"predicted_{s}" in pred_df.columns for s in subjects):
        ax.plot(subjects, predicted_y, marker="s", linewidth=2.5, color="#2563eb", label="Predicted")
    ax.set_title(f"Performance Trend: {student_name}")
    ax.set_ylabel("Marks")
    ax.set_xlabel("Subject")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)


def _plot_performance_distribution(pred_df: pd.DataFrame) -> None:
    if pred_df.empty or "performer_level" not in pred_df.columns:
        st.info("Performance distribution chart will appear once predictions are available.")
        return

    counts = pred_df["performer_level"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Performance Distribution (High / Medium / Low)")
    st.pyplot(fig)


def _recommendation_lookup(recommendations: List[Dict]) -> Dict[str, Dict]:
    lookup: Dict[str, Dict] = {}
    for rec in recommendations:
        key = str(rec.get("Name", "")).strip()
        if key:
            lookup[key] = rec
    return lookup


def _show_metrics(pred_df: pd.DataFrame) -> None:
    if pred_df.empty:
        return

    total = len(pred_df)
    high_risk = int((pred_df["risk_level"] == "High").sum()) if "risk_level" in pred_df else 0
    avg_pred = float(pred_df["predicted_average"].mean()) if "predicted_average" in pred_df else 0.0
    high_perf = int((pred_df["performer_level"] == "High").sum()) if "performer_level" in pred_df else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", total)
    c2.metric("Average Predicted", f"{avg_pred:.2f}")
    c3.metric("High Risk", high_risk)
    c4.metric("High Performers", high_perf)


def _show_data_table(pred_df: pd.DataFrame) -> None:
    if pred_df.empty:
        st.info("No prediction data yet. Use the Control Center to run prediction first.")
        return

    st.subheader("Student Data Table")
    f1, f2, f3 = st.columns([2, 1, 1])

    with f1:
        name_query = st.text_input("Search by student name", placeholder="Type a name")
    with f2:
        risk_options = ["Low", "Medium", "High"]
        risk_filter = st.multiselect("Risk Filter", options=risk_options, default=[])
    with f3:
        perf_options = ["High", "Medium", "Low"]
        perf_filter = st.multiselect("Performance Filter", options=perf_options, default=[])

    table_df = pred_df.copy()

    if name_query:
        table_df = table_df[
            table_df["Name"].astype(str).str.contains(name_query, case=False, na=False)
        ]

    if risk_filter and "risk_level" in table_df.columns:
        table_df = table_df[table_df["risk_level"].isin(risk_filter)]

    if perf_filter and "performer_level" in table_df.columns:
        table_df = table_df[table_df["performer_level"].isin(perf_filter)]

    st.caption("Tip: click column headers to sort.")
    st.dataframe(table_df, use_container_width=True, hide_index=True)


def _show_student_analysis(pred_df: pd.DataFrame, recommendations: List[Dict]) -> None:
    if pred_df.empty:
        st.info("No student analysis available yet. Run prediction first.")
        return

    names = pred_df["Name"].astype(str).tolist()
    selected_name = st.selectbox("Select Student", options=names, index=0)

    subjects = _subject_columns_from_predictions(pred_df)
    _plot_performance_line(pred_df, selected_name, subjects)

    student_row = pred_df[pred_df["Name"] == selected_name].iloc[0]
    rec_map = _recommendation_lookup(recommendations)
    rec = rec_map.get(selected_name, {})

    st.subheader("Individual Analysis")
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Academic Snapshot")
        st.write(f"Predicted Average: {student_row.get('predicted_average', 0):.2f}")
        st.write(f"Risk Level: {student_row.get('risk_level', 'Unknown')}")
        st.write(f"Performer Level: {student_row.get('performer_level', 'Unknown')}")
        weak = student_row.get("weak_subjects", [])
        weak_text = ", ".join(weak) if isinstance(weak, list) and weak else "None"
        st.write(f"Weak Subjects: {weak_text}")

    with right:
        st.markdown("### AI Recommendation")
        st.write(rec.get("recommendation", "Run recommendations to see personalized guidance."))
        st.write(f"NLP Insight: {rec.get('nlp_summary', 'N/A')}")

    pred_cols = _predicted_columns(pred_df)
    if pred_cols:
        st.markdown("### Predicted Marks")
        marks_data = {col: student_row.get(col, 0) for col in pred_cols}
        marks_df = pd.DataFrame({"Subject": marks_data.keys(), "Predicted Marks": marks_data.values()})
        st.dataframe(marks_df, use_container_width=True, hide_index=True)


st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --bg-soft: #f0f9ff;
        --bg-white: #ffffff;
        --blue-strong: #1d4ed8;
        --blue-mid: #0ea5e9;
        --blue-light: #bae6fd;
        --text-main: #0f172a;
        --text-soft: #334155;
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #e0f2fe 0%, #f8fcff 40%, #ffffff 100%);
    }

    [data-testid="stAppViewContainer"] .main {
        padding-top: 0.2rem;
    }

    .block-container {
        /* Prevent title overlap with Streamlit top toolbar/header area */
        padding-top: 3.4rem;
        padding-bottom: 1rem;
    }

    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: var(--text-main);
        margin-top: 0.2rem;
        margin-bottom: 0.3rem;
        animation: fadeSlide 0.7s ease;
    }

    .dashboard-subtitle {
        color: var(--text-soft);
        margin-bottom: 0.9rem;
        animation: fadeSlide 1s ease;
    }

    .vibe-chip {
        display: inline-block;
        background: linear-gradient(90deg, #38bdf8, #3b82f6);
        color: #ffffff;
        border-radius: 999px;
        padding: 0.35rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 1rem;
        box-shadow: 0 10px 20px rgba(14, 165, 233, 0.25);
        animation: pulseGlow 2.2s infinite ease-in-out;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 80%);
        border-right: 1px solid #dbeafe;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-white);
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 8px 18px rgba(14, 165, 233, 0.08);
    }

    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 24px rgba(37, 99, 235, 0.28);
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #dbeafe;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 18px rgba(14, 165, 233, 0.08);
    }

    @keyframes fadeSlide {
        from {opacity: 0; transform: translateY(8px);} 
        to {opacity: 1; transform: translateY(0);} 
    }

    @keyframes pulseGlow {
        0% {transform: scale(1); box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);} 
        50% {transform: scale(1.03); box-shadow: 0 14px 24px rgba(37, 99, 235, 0.30);} 
        100% {transform: scale(1); box-shadow: 0 8px 18px rgba(37, 99, 235, 0.22);} 
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="dashboard-title">AI Student Performance Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dashboard-subtitle">Upload, analyze, and monitor student performance with predictions and AI recommendations.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="vibe-chip">Vibrant Mode: Light Blue + White</div>', unsafe_allow_html=True)

_init_state()

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Control Center", "Data Explorer", "Performance Insights", "Student Analysis"],
    )

    st.divider()
    st.subheader("Backend")
    st.write(f"API URL: {API_URL}")
    if st.button("Check API", use_container_width=True):
        with st.spinner("Checking API status..."):
            try:
                health = _api_get("/")
                st.success(health.get("message", "API reachable"))
            except Exception as exc:
                st.error(f"API not reachable: {exc}")

    st.divider()
    uploaded = st.file_uploader("Upload CSV", type=["csv"])


if page == "Control Center":
    st.header("Control Center")
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Upload + Train", use_container_width=True):
            if uploaded is None:
                st.warning("Please upload a CSV file first.")
            else:
                with st.spinner("Uploading file and training model..."):
                    try:
                        result = _api_upload(uploaded)
                        st.success(result.get("message", "Upload complete"))
                        st.session_state["upload_preview"] = result.get("preview", [])
                        st.session_state["subject_columns"] = result.get("subject_columns", [])
                        st.session_state["predictions"] = []
                        st.session_state["recommendations"] = []
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")

    with b2:
        if st.button("Predict", use_container_width=True):
            with st.spinner("Generating predictions..."):
                try:
                    result = _api_get("/predict")
                    st.session_state["predictions"] = result.get("predictions", [])
                    st.success(result.get("message", "Prediction complete"))
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with b3:
        if st.button("Recommend", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                try:
                    result = _api_get("/recommend", timeout=600)
                    st.session_state["recommendations"] = result.get("recommendations", [])
                    st.success(result.get("message", "Recommendations ready"))
                except Exception as exc:
                    st.error(f"Recommendation failed: {exc}")

    if st.session_state["upload_preview"]:
        st.subheader("Uploaded Data Preview")
        st.dataframe(pd.DataFrame(st.session_state["upload_preview"]), use_container_width=True)
        if st.session_state["subject_columns"]:
            st.caption(f"Detected subject columns: {st.session_state['subject_columns']}")


pred_df = pd.DataFrame(st.session_state["predictions"])
recs = st.session_state["recommendations"]

if page == "Data Explorer":
    st.header("Data Explorer")
    _show_data_table(pred_df)

if page == "Performance Insights":
    st.header("Performance Insights")
    _show_metrics(pred_df)

    subjects = _subject_columns_from_predictions(pred_df)
    c1, c2 = st.columns(2)
    with c1:
        _plot_subject_bar(pred_df, subjects)
    with c2:
        _plot_performance_distribution(pred_df)

if page == "Student Analysis":
    st.header("Student Analysis")
    _show_student_analysis(pred_df, recs)

    if recs:
        st.subheader("Recommendation Records")
        st.dataframe(pd.DataFrame(recs), use_container_width=True, hide_index=True)
