import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PID_FILE = PROJECT_ROOT / ".flask_backend.pid"
BACKEND_LOG = PROJECT_ROOT / "flask_backend.log"

API_BASE = os.getenv("FLASK_API_URL", "http://127.0.0.1:5000")
BACKEND_AUTOSTART = os.getenv("BACKEND_AUTOSTART", "true").strip().lower() in {"1", "true", "yes", "on"}
FLASK_START_CMD = os.getenv("FLASK_START_CMD", "").strip()


st.set_page_config(page_title="Clinical Drug Recommendation + Sentiment", layout="wide")
st.title("Patient Condition-Aware Drug Recommendation and Clinical Review Sentiment")
st.caption("Flask backend + Hugging Face API inference (no local model download).")

col1, col2 = st.columns(2)


def api_is_available() -> bool:
    try:
        res = requests.get(f"{API_BASE}/health", timeout=2)
        return res.status_code == 200
    except Exception:
        return False


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _write_pid(pid: int) -> None:
    PID_FILE.write_text(str(pid), encoding="utf-8")


def _backend_cmd() -> list[str]:
    if FLASK_START_CMD:
        return shlex.split(FLASK_START_CMD)
    return [sys.executable, "-m", "api.app"]


def ensure_backend() -> tuple[bool, str]:
    if api_is_available():
        return True, "Flask backend connected."

    if not BACKEND_AUTOSTART:
        return False, "Backend auto-start is disabled."

    existing_pid = _read_pid()
    if existing_pid and _is_pid_running(existing_pid):
        for _ in range(10):
            if api_is_available():
                return True, f"Flask backend connected (PID {existing_pid})."
            time.sleep(0.4)
        return False, f"Backend process exists (PID {existing_pid}) but API is still unreachable."

    cmd = _backend_cmd()
    BACKEND_LOG.touch(exist_ok=True)
    with BACKEND_LOG.open("a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )

    _write_pid(proc.pid)

    for _ in range(20):
        if api_is_available():
            return True, f"Started Flask backend automatically (PID {proc.pid})."
        if proc.poll() is not None:
            break
        time.sleep(0.4)

    return False, (
        f"Tried to auto-start backend with command: {' '.join(cmd)}. "
        f"Check backend logs in: {BACKEND_LOG}"
    )


def show_backend_down_message(reason: str) -> None:
    st.error("Flask backend is not running or not reachable.")
    st.caption(reason)
    st.code(
        "conda activate drug311\ncd /home/rohini/Downloads/CIET/drug\npython -m api.app",
        language="bash",
    )


def render_recommendation_result(data: dict) -> None:
    st.success(f"Confidence: {data.get('confidence', 0)}")
    st.write("### Top Recommendations")
    st.dataframe(data.get("top_recommendations", []), use_container_width=True)
    st.write("### Excluded Options")
    st.dataframe(data.get("excluded_options", []), use_container_width=True)
    st.write("### Clinical Review")
    st.info(data.get("clinical_review", ""))
    if data.get("parsed_input"):
        st.write("### Parsed Input (Dynamic Mode)")
        st.json(data["parsed_input"])
    if data.get("disclaimer"):
        st.warning(data["disclaimer"])


backend_ok, backend_msg = ensure_backend()
if backend_ok:
    st.caption(backend_msg)

with col1:
    st.subheader("Drug Recommendation")
    if not backend_ok:
        show_backend_down_message(backend_msg)
    tab1, tab2 = st.tabs(["Structured Input", "Dynamic Text / Upload Reports"])

    with tab1:
        condition = st.selectbox(
            "Condition",
            ["hypertension", "type_2_diabetes", "major_depressive_disorder", "asthma"],
            key="structured_condition",
        )
        age = st.number_input("Age", min_value=0, max_value=120, value=45, key="structured_age")
        pregnant = st.checkbox("Pregnant", value=False, key="structured_pregnant")
        renal_impairment = st.checkbox("Renal impairment", value=False, key="structured_renal")
        liver_impairment = st.checkbox("Liver impairment", value=False, key="structured_liver")
        allergies = st.text_input("Allergies (comma-separated)", "", key="structured_allergies")
        comorbidities = st.text_input("Comorbidities (comma-separated)", "", key="structured_comorbidities")
        current_meds = st.text_input("Current medications (comma-separated)", "", key="structured_meds")

        if st.button("Get Recommendations", type="primary", key="structured_submit"):
            if not backend_ok:
                show_backend_down_message(backend_msg)
                st.stop()

            payload = {
                "condition": condition,
                "age": int(age),
                "pregnant": pregnant,
                "allergies": [x.strip() for x in allergies.split(",") if x.strip()],
                "comorbidities": [x.strip() for x in comorbidities.split(",") if x.strip()],
                "current_medications": [x.strip() for x in current_meds.split(",") if x.strip()],
                "renal_impairment": renal_impairment,
                "liver_impairment": liver_impairment,
            }

            try:
                res = requests.post(f"{API_BASE}/recommend", json=payload, timeout=30)
                if res.status_code == 200:
                    render_recommendation_result(res.json())
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"API error: {e}")

    with tab2:
        description = st.text_area(
            "Describe patient condition / history / labs",
            "56 year old with high blood pressure and CKD stage 3. eGFR: 48, Creatinine: 1.8. Current medications: Aspirin. Allergies: penicillin.",
            height=170,
            key="dynamic_description",
        )
        uploaded_files = st.file_uploader(
            "Upload reports (.txt, .csv, .json, .pdf)",
            type=["txt", "csv", "json", "pdf"],
            accept_multiple_files=True,
            key="dynamic_files",
        )

        st.write("Optional manual overrides")
        d_condition = st.text_input("Condition override (optional)", "", key="dynamic_condition")
        d_age = st.number_input("Age override", min_value=0, max_value=120, value=45, key="dynamic_age")
        d_pregnant = st.checkbox("Pregnant override", value=False, key="dynamic_pregnant")
        d_renal = st.checkbox("Renal impairment override", value=False, key="dynamic_renal")
        d_liver = st.checkbox("Liver impairment override", value=False, key="dynamic_liver")
        d_allergies = st.text_input("Allergies override (comma-separated)", "", key="dynamic_allergies")
        d_comorbidities = st.text_input("Comorbidities override (comma-separated)", "", key="dynamic_comorbidities")
        d_meds = st.text_input("Current medications override (comma-separated)", "", key="dynamic_meds")

        if st.button("Analyze Dynamic Intake", type="primary", key="dynamic_submit"):
            if not backend_ok:
                show_backend_down_message(backend_msg)
                st.stop()

            data = {
                "description": description,
                "condition": d_condition.strip(),
                "age": str(int(d_age)),
                "pregnant": str(d_pregnant).lower(),
                "renal_impairment": str(d_renal).lower(),
                "liver_impairment": str(d_liver).lower(),
                "allergies": d_allergies,
                "comorbidities": d_comorbidities,
                "current_medications": d_meds,
            }

            files = []
            for file in uploaded_files or []:
                files.append(
                    (
                        "reports",
                        (
                            file.name,
                            file.getvalue(),
                            file.type or "application/octet-stream",
                        ),
                    )
                )

            try:
                res = requests.post(f"{API_BASE}/recommend_dynamic", data=data, files=files, timeout=90)
                if res.status_code == 200:
                    render_recommendation_result(res.json())
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"API error: {e}")

with col2:
    st.subheader("Clinical Review Sentiment Analysis")
    model_name = st.text_input(
        "HF sentiment model (optional)",
        value=os.getenv("HF_SENTIMENT_MODEL", "siebert/sentiment-roberta-large-english"),
    )
    reviews_raw = st.text_area(
        "Paste one review per line",
        "Patient tolerated metformin well with better fasting glucose.\nSevere nausea after first week.",
        height=220,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not backend_ok:
            show_backend_down_message(backend_msg)
            st.stop()

        reviews = [line.strip() for line in reviews_raw.splitlines() if line.strip()]
        payload = {"reviews": reviews, "model": model_name}

        try:
            res = requests.post(f"{API_BASE}/sentiment", json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                st.write("### Aggregate")
                st.json(data["aggregate"])
                st.write("### Per Review")
                st.dataframe(data["per_review"], use_container_width=True)
            else:
                st.error(res.text)
        except Exception as e:
            st.error(f"API error: {e}")

st.divider()
st.write("### API Config")
st.code(
    json.dumps(
        {
            "FLASK_API_URL": API_BASE,
            "BACKEND_AUTOSTART": BACKEND_AUTOSTART,
            "FLASK_START_CMD": FLASK_START_CMD or f"{sys.executable} -m api.app",
        },
        indent=2,
    ),
    language="json",
)
