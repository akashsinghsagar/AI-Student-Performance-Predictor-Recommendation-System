from typing import Dict, List

from app.utils.logger import get_logger

try:
    from transformers import pipeline
except Exception:  # pragma: no cover
    pipeline = None


logger = get_logger(__name__)


class PerformanceNLPAnalyzer:
    def __init__(self) -> None:
        self.unmasker = None
        self.max_transformer_rows = 120
        self.large_batch_threshold = 500

        if pipeline is not None:
            try:
                self.unmasker = pipeline("fill-mask", model="distilbert-base-uncased")
                logger.info("DistilBERT fill-mask pipeline initialized.")
            except Exception as exc:
                logger.warning("Transformer initialization failed; using rule-based NLP: %s", exc)
                self.unmasker = None

    @staticmethod
    def build_performance_text(student_row: Dict) -> str:
        name = student_row.get("Name", "Student")
        performer = student_row.get("performer_level", "Unknown")
        weak_subjects = student_row.get("weak_subjects", [])
        improvement = float(student_row.get("improvement_rate", 0))

        weak_text = ", ".join(weak_subjects) if weak_subjects else "none"

        trend_text = "stable"
        if improvement > 5:
            trend_text = "improving"
        elif improvement < -5:
            trend_text = "declining"

        return (
            f"{name} is a {performer} performer, weak in {weak_text}, "
            f"and trend is {trend_text}."
        )

    @staticmethod
    def _classify_rule_based(item: Dict) -> Dict:
        risk = str(item.get("risk_level", "Low")).lower()
        improvement = float(item.get("improvement_rate", 0.0))
        performer = str(item.get("performer_level", "Medium")).lower()

        if risk == "high" or performer == "low" or improvement < -5:
            return {
                "label": "NEGATIVE",
                "score": 0.65,
                "summary": "Potential performance concern detected.",
            }

        if risk == "low" and (performer == "high" or improvement > 2):
            return {
                "label": "POSITIVE",
                "score": 0.65,
                "summary": "Positive learning momentum detected.",
            }

        return {
            "label": "NEUTRAL",
            "score": 0.5,
            "summary": "Neutral learning signal detected.",
        }

    def classify_text(self, text: str) -> Dict:
        if not self.unmasker:
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "summary": "Transformer model unavailable, using neutral fallback insight.",
            }

        try:
            masked_text = f"{text} Overall performance is {self.unmasker.tokenizer.mask_token}."
            candidates = self.unmasker(masked_text, top_k=10)
        except Exception as exc:
            logger.warning("Transformer inference failed: %s", exc)
            return {
                "label": "NEUTRAL",
                "score": 0.0,
                "summary": "Transformer inference failed, using neutral fallback insight.",
            }

        positive_words = {"good", "great", "excellent", "strong", "better", "improving"}
        negative_words = {"poor", "weak", "bad", "worse", "failing", "low"}

        positive_score = 0.0
        negative_score = 0.0

        for item in candidates:
            token = item.get("token_str", "").strip().lower()
            token_score = float(item.get("score", 0.0))
            if token in positive_words:
                positive_score += token_score
            if token in negative_words:
                negative_score += token_score

        if positive_score > negative_score + 0.01:
            return {
                "label": "POSITIVE",
                "score": round(positive_score, 4),
                "summary": "Positive learning momentum detected.",
            }
        if negative_score > positive_score + 0.01:
            return {
                "label": "NEGATIVE",
                "score": round(negative_score, 4),
                "summary": "Potential performance concern detected.",
            }

        return {
            "label": "NEUTRAL",
            "score": round(max(positive_score, negative_score), 4),
            "summary": "Neutral learning signal detected.",
        }

    def enrich(self, recommendations: List[Dict]) -> List[Dict]:
        enriched: List[Dict] = []
        use_transformer = bool(self.unmasker)
        total = len(recommendations)
        transformer_limit = self.max_transformer_rows

        if total > self.large_batch_threshold:
            use_transformer = False
            transformer_limit = 0

        for idx, item in enumerate(recommendations):
            perf_text = self.build_performance_text(item)

            if use_transformer and idx < transformer_limit:
                clf = self.classify_text(perf_text)
            else:
                clf = self._classify_rule_based(item)

            if total > transformer_limit and idx == transformer_limit:
                use_transformer = False

            new_item = dict(item)
            new_item["performance_text"] = perf_text
            new_item["nlp_label"] = clf["label"]
            new_item["nlp_score"] = clf["score"]
            new_item["nlp_summary"] = clf["summary"]
            enriched.append(new_item)

        return enriched
