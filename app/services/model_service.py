import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

from app.utils.logger import get_logger


logger = get_logger(__name__)


NON_SUBJECT_COLUMNS = {
    "attendance",
    "avg_mark",
    "std_mark",
    "improvement_rate",
    "predicted_average",
    "previous_avg",
    "prev_avg",
    "last_sem_avg",
    "prev_sem_avg",
    "id",
    "absence_days",
    "weekly_self_study_hours",
}


class StudentPerformanceModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = str(model_path)
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_columns: List[str] = []
        self.subject_columns: List[str] = []
        self.target_mode = "synthetic"

        if os.path.exists(self.model_path):
            self.load()
            logger.info("Loaded model from %s", self.model_path)

    @staticmethod
    def _ensure_name_column(df: pd.DataFrame) -> pd.DataFrame:
        if "Name" not in df.columns:
            df = df.copy()
            if "first_name" in df.columns and "last_name" in df.columns:
                df["Name"] = (
                    df["first_name"].fillna("") + " " + df["last_name"].fillna("")
                ).str.strip()
                df["Name"] = df["Name"].replace("", np.nan).fillna("Student")
            else:
                df["Name"] = [f"Student_{i + 1}" for i in range(len(df))]
        return df

    @staticmethod
    def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        cast_df = df.copy()
        for col in cast_df.columns:
            if col == "Name":
                continue
            cast_df[col] = pd.to_numeric(cast_df[col], errors="coerce")
        return cast_df

    @staticmethod
    def _find_previous_average_column(df: pd.DataFrame) -> str:
        for col in ["previous_avg", "prev_avg", "last_sem_avg", "prev_sem_avg"]:
            if col in df.columns:
                return col
        return ""

    def _get_subject_columns(self, df: pd.DataFrame) -> List[str]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        candidate_cols = [
            col
            for col in numeric_cols
            if col.lower() not in NON_SUBJECT_COLUMNS and not col.lower().startswith("next_")
        ]

        score_like_cols = [
            col
            for col in candidate_cols
            if re.search(r"(score|marks?)", col.lower())
        ]
        if score_like_cols:
            return score_like_cols

        fallback_cols: List[str] = []
        for col in candidate_cols:
            non_null_ratio = float(df[col].notna().mean())
            if non_null_ratio >= 0.5:
                fallback_cols.append(col)

        return fallback_cols

    def _create_features(
        self, df: pd.DataFrame, subject_cols: List[str], fit: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        work_df = df.copy()

        for col in subject_cols:
            median_val = work_df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            work_df[col] = work_df[col].fillna(median_val)

        if "attendance" not in work_df.columns:
            if "absence_days" in work_df.columns:
                work_df["attendance"] = (100 - (work_df["absence_days"].fillna(0) * 2)).clip(0, 100)
            else:
                work_df["attendance"] = 75.0

        work_df["attendance"] = work_df["attendance"].fillna(work_df["attendance"].mean())
        work_df["attendance"] = work_df["attendance"].fillna(75.0)

        prev_avg_col = self._find_previous_average_column(work_df)
        work_df["avg_mark"] = work_df[subject_cols].mean(axis=1)
        work_df["std_mark"] = work_df[subject_cols].std(axis=1).fillna(0.0)

        if prev_avg_col:
            baseline = work_df[prev_avg_col].replace(0, np.nan)
            work_df["improvement_rate"] = (
                ((work_df["avg_mark"] - work_df[prev_avg_col]) / baseline) * 100
            ).fillna(0.0)
        else:
            work_df["improvement_rate"] = 0.0

        if fit:
            normalized = self.scaler.fit_transform(work_df[subject_cols])
        else:
            normalized = self.scaler.transform(work_df[subject_cols])

        norm_cols = [f"{col}_norm" for col in subject_cols]
        norm_df = pd.DataFrame(normalized, columns=norm_cols, index=work_df.index)
        work_df = pd.concat([work_df, norm_df], axis=1)

        feature_cols = norm_cols + ["attendance", "avg_mark", "std_mark", "improvement_rate"]
        features = work_df[feature_cols]

        return work_df, features

    def preprocess(self, raw_df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self._ensure_name_column(raw_df)
        df = self._to_numeric(df)

        subject_cols = self._get_subject_columns(df)
        if fit:
            self.subject_columns = subject_cols
        else:
            subject_cols = self.subject_columns

        if not subject_cols:
            raise ValueError("No subject/marks columns were found in the input data.")

        missing_cols = [c for c in subject_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required subject columns: {missing_cols}")

        processed_df, features = self._create_features(df, subject_cols, fit=fit)

        if fit:
            self.feature_columns = features.columns.tolist()
        else:
            features = features[self.feature_columns]

        return processed_df, features

    def _build_targets(self, df: pd.DataFrame, subject_cols: List[str]) -> pd.DataFrame:
        next_cols = [f"next_{c}" for c in subject_cols]
        has_real_targets = all(c in df.columns for c in next_cols)

        if has_real_targets:
            self.target_mode = "real"
            target_df = df[next_cols].copy()
            for col in next_cols:
                target_df[col] = pd.to_numeric(target_df[col], errors="coerce")
                target_df[col] = target_df[col].fillna(target_df[col].median())
            return target_df

        self.target_mode = "synthetic"
        synthetic_df = pd.DataFrame(index=df.index)

        attendance_adjustment = (df["attendance"] - 75.0) * 0.1
        improvement_adjustment = df["improvement_rate"] * 0.1

        for col in subject_cols:
            synthetic_df[f"next_{col}"] = (
                df[col] + attendance_adjustment + improvement_adjustment
            ).clip(0, 100)

        return synthetic_df

    def train(self, train_df: pd.DataFrame) -> None:
        processed_df, features = self.preprocess(train_df, fit=True)
        targets = self._build_targets(processed_df, self.subject_columns)

        base_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
        )
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(features, targets)

        self.save()

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model is not trained. Upload and train first.")

        processed_df, features = self.preprocess(input_df, fit=False)
        pred_values = self.model.predict(features)

        pred_cols = [f"predicted_{col}" for col in self.subject_columns]
        pred_df = pd.DataFrame(pred_values, columns=pred_cols, index=processed_df.index)

        output_df = processed_df[
            ["Name"] + self.subject_columns + ["attendance", "avg_mark", "improvement_rate"]
        ].copy()
        output_df = pd.concat([output_df, pred_df], axis=1)

        output_df["predicted_average"] = pred_df.mean(axis=1).round(2)
        output_df["performer_level"] = output_df["predicted_average"].apply(self._classify_performer)
        output_df["risk_level"] = output_df.apply(self._classify_risk, axis=1)
        output_df["weak_subjects"] = output_df[self.subject_columns].apply(
            lambda row: [subj for subj, value in row.items() if value < 40], axis=1
        )

        for col in pred_cols:
            output_df[col] = output_df[col].round(2)

        return output_df

    @staticmethod
    def _classify_performer(predicted_average: float) -> str:
        if predicted_average >= 75:
            return "High"
        if predicted_average >= 50:
            return "Medium"
        return "Low"

    @staticmethod
    def _classify_risk(row: pd.Series) -> str:
        avg = row["predicted_average"]
        attendance = row.get("attendance", 75)
        if avg < 45 or attendance < 60:
            return "High"
        if avg < 65 or attendance < 75:
            return "Medium"
        return "Low"

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        payload: Dict[str, object] = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "subject_columns": self.subject_columns,
            "target_mode": self.target_mode,
        }
        joblib.dump(payload, self.model_path)
        logger.info("Model saved to %s", self.model_path)

    def load(self) -> None:
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_columns = payload["feature_columns"]
        self.subject_columns = payload["subject_columns"]
        self.target_mode = payload.get("target_mode", "synthetic")
