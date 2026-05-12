import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pydantic import ValidationError

try:
    from .services.hf_client import HFClient
    from .services.intake_parser import (
        SUPPORTED_CONDITIONS,
        build_patient_from_dynamic_input,
        parse_report_file,
    )
    from .services.recommender import DrugRecommender
    from .services.schemas import PatientProfile, SentimentRequest
except ImportError:
    from services.hf_client import HFClient
    from services.intake_parser import (
        SUPPORTED_CONDITIONS,
        build_patient_from_dynamic_input,
        parse_report_file,
    )
    from services.recommender import DrugRecommender
    from services.schemas import PatientProfile, SentimentRequest

load_dotenv()

app = Flask(__name__)
recommender = DrugRecommender()


@app.get("/")
def index():
    return jsonify(
        {
            "service": "Patient Condition-Aware Drug Recommendation API",
            "status": "running",
            "endpoints": {
                "GET /health": "Health check",
                "POST /recommend": "Structured recommendation input (JSON)",
                "POST /recommend_dynamic": "Dynamic text/report recommendation input (JSON or multipart)",
                "POST /sentiment": "Clinical review sentiment analysis",
            },
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}
    try:
        patient = PatientProfile(**payload)
    except ValidationError as e:
        return jsonify({"error": "validation_failed", "details": e.errors()}), 422

    result = recommender.recommend(patient)
    return jsonify(result)


@app.post("/recommend_dynamic")
def recommend_dynamic():
    manual_overrides: Dict[str, Any] = {}
    description = ""
    report_texts: List[str] = []
    processed_files: List[str] = []

    if request.content_type and (
        "multipart/form-data" in request.content_type or "application/x-www-form-urlencoded" in request.content_type
    ):
        form = request.form
        description = str(form.get("description", ""))

        raw_age = form.get("age")
        if raw_age:
            try:
                manual_overrides["age"] = int(raw_age)
            except ValueError:
                return jsonify({"error": "validation_failed", "details": "age must be an integer"}), 422

        for key in ["condition"]:
            if form.get(key):
                manual_overrides[key] = str(form.get(key))

        for key in ["pregnant", "renal_impairment", "liver_impairment"]:
            if form.get(key) is not None:
                manual_overrides[key] = str(form.get(key)).lower() in ("1", "true", "yes", "on")

        for key in ["allergies", "comorbidities", "current_medications"]:
            raw = str(form.get(key, ""))
            if raw.strip():
                manual_overrides[key] = [x.strip().lower() for x in raw.split(",") if x.strip()]

        for file in request.files.getlist("reports"):
            if not file or not file.filename:
                continue
            content = parse_report_file(file.filename, file.read())
            if content.strip():
                report_texts.append(content)
                processed_files.append(file.filename)
    else:
        payload = request.get_json(silent=True) or {}
        description = str(payload.get("description", ""))
        manual_overrides = payload.get("patient_profile", {})
        report_texts = payload.get("report_texts", [])

    try:
        patient, trace = build_patient_from_dynamic_input(
            description=description,
            report_texts=report_texts,
            manual_overrides=manual_overrides,
        )
    except ValidationError as e:
        return jsonify({"error": "validation_failed", "details": e.errors()}), 422

    if not patient.condition:
        supported = sorted(list(SUPPORTED_CONDITIONS.keys()))
        return (
            jsonify(
                {
                    "error": "condition_not_detected",
                    "details": "Could not detect a supported condition from description/reports.",
                    "supported_conditions": supported,
                }
            ),
            422,
        )

    result = recommender.recommend(patient)
    result["parsed_input"] = {
        "patient_profile": patient.model_dump(),
        "trace": trace,
        "files_used": processed_files,
    }
    return jsonify(result)


@app.post("/sentiment")
def sentiment():
    payload = request.get_json(silent=True) or {}
    try:
        req = SentimentRequest(**payload)
    except ValidationError as e:
        return jsonify({"error": "validation_failed", "details": e.errors()}), 422

    try:
        client = HFClient()
        per_review = []
        for text in req.reviews:
            if not text.strip():
                continue
            sentiment_result = client.classify_sentiment(text=text[:2000], model=req.model)
            per_review.append({"text": text, **sentiment_result})

        aggregate = client.aggregate(per_review)
        return jsonify(
            {
                "review_count": len(per_review),
                "per_review": per_review,
                "aggregate": aggregate,
            }
        )
    except Exception as e:
        return jsonify({"error": "hf_inference_failed", "details": str(e)}), 500


@app.errorhandler(404)
def handle_not_found(_):
    return (
        jsonify(
            {
                "error": "not_found",
                "details": "Route not found.",
                "hint": "Use GET / for API index and available endpoints.",
            }
        ),
        404,
    )


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
