import json
import time
import traceback
from pathlib import Path

import requests


def run() -> int:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts_file = logs_dir / "latest_run.ts"
    if ts_file.exists():
        ts = ts_file.read_text(encoding="utf-8").strip()
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        ts_file.write_text(ts, encoding="utf-8")

    base = "http://127.0.0.1:5000"
    out_json = logs_dir / f"api_test_{ts}.json"
    out_log = logs_dir / f"api_test_{ts}.log"

    for _ in range(20):
        try:
            response = requests.get(f"{base}/health", timeout=2)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)

    cases = [
        {
            "name": "health_ok",
            "method": "GET",
            "path": "/health",
            "expect_codes": [200],
            "expect_contains": ["status"],
        },
        {
            "name": "index_ok",
            "method": "GET",
            "path": "/",
            "expect_codes": [200],
            "expect_contains": ["service", "endpoints"],
        },
        {
            "name": "recommend_valid_hypertension",
            "method": "POST",
            "path": "/recommend",
            "json": {
                "condition": "hypertension",
                "age": 56,
                "pregnant": False,
                "allergies": ["penicillin"],
                "comorbidities": ["chronic_kidney_disease"],
                "current_medications": ["Aspirin"],
                "renal_impairment": True,
                "liver_impairment": False,
            },
            "expect_codes": [200],
            "expect_contains": ["top_recommendations", "excluded_options", "confidence"],
        },
        {
            "name": "recommend_validation_missing_condition",
            "method": "POST",
            "path": "/recommend",
            "json": {"age": 56, "pregnant": False},
            "expect_codes": [422],
            "expect_contains": ["validation_failed"],
        },
        {
            "name": "recommend_unknown_condition",
            "method": "POST",
            "path": "/recommend",
            "json": {
                "condition": "rare_unknown_condition_xyz",
                "age": 33,
                "pregnant": False,
                "allergies": [],
                "comorbidities": [],
                "current_medications": [],
                "renal_impairment": False,
                "liver_impairment": False,
            },
            "expect_codes": [200],
            "expect_contains": ["top_recommendations", "clinical_review", "confidence"],
        },
        {
            "name": "recommend_dynamic_json_detected",
            "method": "POST",
            "path": "/recommend_dynamic",
            "json": {
                "description": (
                    "56 year old with high blood pressure and CKD stage 3. eGFR: 48, "
                    "creatinine: 1.8. current medications: aspirin. allergies: penicillin."
                ),
                "patient_profile": {},
                "report_texts": [],
            },
            "expect_codes": [200],
            "expect_contains": ["parsed_input", "top_recommendations", "condition"],
        },
        {
            "name": "recommend_dynamic_multipart_txt",
            "method": "POST",
            "path": "/recommend_dynamic",
            "data": {
                "description": "Patient with asthma and wheezing, age: 28",
                "current_medications": "albuterol",
            },
            "files": [
                (
                    "reports",
                    (
                        "report.txt",
                        b"History includes recurrent UTI. creatinine: 0.9",
                        "text/plain",
                    ),
                )
            ],
            "expect_codes": [200],
            "expect_contains": ["parsed_input", "files_used"],
        },
        {
            "name": "recommend_dynamic_condition_not_detected",
            "method": "POST",
            "path": "/recommend_dynamic",
            "json": {
                "description": "Patient reports mild fatigue and insomnia only.",
                "patient_profile": {},
                "report_texts": [],
            },
            "expect_codes": [422],
            "expect_contains": ["condition_not_detected"],
        },
        {
            "name": "recommend_dynamic_bad_age_multipart",
            "method": "POST",
            "path": "/recommend_dynamic",
            "data": {"description": "known hypertension patient", "age": "abc"},
            "expect_codes": [422],
            "expect_contains": ["age must be an integer"],
        },
        {
            "name": "sentiment_validation_empty_reviews",
            "method": "POST",
            "path": "/sentiment",
            "json": {"reviews": []},
            "expect_codes": [422],
            "expect_contains": ["validation_failed"],
        },
        {
            "name": "sentiment_runtime",
            "method": "POST",
            "path": "/sentiment",
            "json": {
                "reviews": [
                    "Patient tolerated metformin well with improved glucose.",
                    "Experienced significant nausea and fatigue.",
                ]
            },
            "expect_codes": [200, 500],
            "expect_contains": [],
        },
        {
            "name": "not_found_route",
            "method": "GET",
            "path": "/does-not-exist",
            "expect_codes": [404],
            "expect_contains": ["not_found"],
        },
    ]

    results = []
    passed = 0
    for case in cases:
        result = {
            "name": case["name"],
            "method": case["method"],
            "path": case["path"],
            "expected_codes": case["expect_codes"],
        }
        try:
            kwargs = {"timeout": 120}
            if "json" in case:
                kwargs["json"] = case["json"]
            if "data" in case:
                kwargs["data"] = case["data"]
            if "files" in case:
                kwargs["files"] = case["files"]

            response = requests.request(case["method"], f"{base}{case['path']}", **kwargs)
            text = response.text
            result["status_code"] = response.status_code
            result["status_ok"] = response.status_code in case["expect_codes"]
            result["response_preview"] = text[:1000]

            try:
                response_json = response.json()
            except Exception:
                response_json = None
            result["response_json"] = response_json

            missing_tokens = [token for token in case.get("expect_contains", []) if token not in text]
            result["missing_tokens"] = missing_tokens
            result["contains_ok"] = len(missing_tokens) == 0

            ok = bool(result["status_ok"] and result["contains_ok"])

            if ok and case["name"] == "recommend_valid_hypertension" and isinstance(response_json, dict):
                if not response_json.get("top_recommendations"):
                    ok = False
                    result["extra_check"] = "top_recommendations should not be empty"

            if ok and case["name"] == "recommend_dynamic_json_detected" and isinstance(response_json, dict):
                condition = (((response_json.get("parsed_input") or {}).get("patient_profile") or {}).get("condition"))
                if condition != "hypertension":
                    ok = False
                    result["extra_check"] = f"expected detected condition hypertension, got {condition}"

            result["pass"] = ok
            if ok:
                passed += 1
        except Exception as exc:
            result["pass"] = False
            result["exception"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()

        results.append(result)

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [f"API TEST SUMMARY: {passed}/{len(results)} passed"]
    for result in results:
        status = "PASS" if result.get("pass") else "FAIL"
        lines.append(
            f"- [{status}] {result['name']} code={result.get('status_code')} "
            f"expected={result.get('expected_codes')}"
        )
        if not result.get("pass"):
            if result.get("missing_tokens"):
                lines.append(f"    missing_tokens={result['missing_tokens']}")
            if result.get("extra_check"):
                lines.append(f"    extra_check={result['extra_check']}")
            if result.get("exception"):
                lines.append(f"    exception={result['exception']}")
    out_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"WROTE {out_json}")
    print(f"WROTE {out_log}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
