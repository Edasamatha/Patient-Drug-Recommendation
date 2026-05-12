import os
from typing import Dict, List

from huggingface_hub import InferenceClient


class HFClient:
    def __init__(self) -> None:
        token = os.getenv("HF_API_KEY")
        if not token:
            raise ValueError("HF_API_KEY not set. Add it in environment or .env file.")

        # Strong default with robust fallback chain for API availability.
        self.default_model = os.getenv(
            "HF_SENTIMENT_MODEL",
            "siebert/sentiment-roberta-large-english",
        )
        fallback_raw = os.getenv(
            "HF_SENTIMENT_FALLBACK_MODELS",
            "cardiffnlp/twitter-roberta-base-sentiment-latest,distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        )
        self.fallback_models = [x.strip() for x in fallback_raw.split(",") if x.strip()]
        self.router_base = os.getenv("HF_ROUTER_BASE", "https://router.huggingface.co/hf-inference").rstrip("/")
        self.client = InferenceClient(token=token)

    def _to_model_target(self, model: str) -> str:
        if model.startswith("http://") or model.startswith("https://"):
            return model
        return f"{self.router_base}/models/{model}"

    def classify_sentiment(self, text: str, model: str | None = None) -> Dict:
        candidates = [model] if model else [self.default_model, *self.fallback_models]
        last_error = None

        for chosen_model in [m for m in candidates if m]:
            try:
                target = self._to_model_target(chosen_model)
                output = self.client.text_classification(text=text, model=target)
                if not output:
                    continue
                best = max(output, key=lambda x: x.score)
                return {
                    "label": str(best.label).upper(),
                    "score": float(best.score),
                    "model": chosen_model,
                    "model_target": target,
                    "all_scores": [{"label": o.label, "score": float(o.score)} for o in output],
                }
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise RuntimeError(f"All sentiment models failed. Last error: {last_error}") from last_error
        return {"label": "UNKNOWN", "score": 0.0, "model": None, "all_scores": []}

    @staticmethod
    def aggregate(results: List[Dict]) -> Dict:
        if not results:
            return {"distribution": {}, "avg_confidence": 0.0}

        distribution: Dict[str, int] = {}
        score_sum = 0.0

        for item in results:
            label = item["label"]
            distribution[label] = distribution.get(label, 0) + 1
            score_sum += float(item.get("score", 0.0))

        return {
            "distribution": distribution,
            "avg_confidence": round(score_sum / len(results), 4),
            "dominant_label": max(distribution, key=distribution.get),
        }
