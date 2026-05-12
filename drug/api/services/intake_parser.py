import io
import json
import re
from typing import Dict, List, Optional, Tuple

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

from .schemas import PatientProfile


SUPPORTED_CONDITIONS = {
    "hypertension": ["hypertension", "high blood pressure", "htn"],
    "type_2_diabetes": ["type 2 diabetes", "t2dm", "diabetes mellitus type 2", "diabetes"],
    "major_depressive_disorder": ["major depressive disorder", "depression", "mdd"],
    "asthma": ["asthma", "bronchial asthma", "wheezing"],
}

COMORBIDITY_KEYWORDS = {
    "chronic_kidney_disease": ["ckd", "chronic kidney disease", "kidney disease"],
    "liver_disease": ["liver disease", "cirrhosis", "hepatitis"],
    "heart_failure": ["heart failure", "hfref", "hfpef"],
    "hyperkalemia": ["hyperkalemia", "high potassium"],
    "bipolar_disorder": ["bipolar"],
    "seizure_disorder": ["seizure disorder", "epilepsy"],
    "eating_disorder": ["eating disorder", "anorexia", "bulimia"],
    "arrhythmia": ["arrhythmia", "afib", "atrial fibrillation"],
    "qt_prolongation": ["qt prolongation", "long qt"],
    "recurrent_uti": ["recurrent uti", "recurrent urinary tract infection"],
    "pancreatitis_history": ["pancreatitis"],
}

ALLERGY_PAT = re.compile(r"(?:allerg(?:y|ies)|allergic to)\b\s*[:\-]?\s*([^\n\.]+)", re.IGNORECASE)
MEDS_PAT = re.compile(
    r"(?:current\s+medications?|current\s+meds?|medications?)\b\s*[:\-]?\s*([^\n\.]+)",
    re.IGNORECASE,
)
AGE_PATTERNS = [
    re.compile(r"\bage\s*[:=]?\s*(\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b", re.IGNORECASE),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_condition(text: str) -> Tuple[Optional[str], Optional[str]]:
    norm = _normalize(text)
    for canonical, aliases in SUPPORTED_CONDITIONS.items():
        for alias in aliases:
            if alias in norm:
                return canonical, alias
    return None, None


def extract_age(text: str) -> Optional[int]:
    for pat in AGE_PATTERNS:
        m = pat.search(text)
        if m:
            age = int(m.group(1))
            if 0 <= age <= 120:
                return age
    return None


def extract_simple_list(pattern: re.Pattern, text: str) -> List[str]:
    m = pattern.search(text)
    if not m:
        return []
    raw = m.group(1)
    return [x.strip().lower() for x in re.split(r",|;|/", raw) if x.strip()]


def extract_comorbidities(text: str) -> List[str]:
    norm = _normalize(text)
    hits: List[str] = []
    for tag, keywords in COMORBIDITY_KEYWORDS.items():
        if any(k in norm for k in keywords):
            hits.append(tag)
    return hits


def extract_labs(text: str) -> Dict[str, float]:
    labs: Dict[str, float] = {}
    patterns = {
        "egfr": re.compile(r"\begfr\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
        "creatinine": re.compile(r"\bcreatinine\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
        "alt": re.compile(r"\balt\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
        "ast": re.compile(r"\bast\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
        "hba1c": re.compile(r"\bhba1c\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    }
    for name, pat in patterns.items():
        m = pat.search(text)
        if m:
            labs[name] = float(m.group(1))
    return labs


def infer_renal_impairment(labs: Dict[str, float], text: str) -> bool:
    if labs.get("egfr", 999) < 60:
        return True
    if labs.get("creatinine", 0) > 1.5:
        return True
    return "renal impairment" in _normalize(text)


def infer_liver_impairment(labs: Dict[str, float], text: str) -> bool:
    if labs.get("alt", 0) >= 120 or labs.get("ast", 0) >= 120:
        return True
    norm = _normalize(text)
    return "liver impairment" in norm or "hepatic impairment" in norm


def parse_report_file(filename: str, data: bytes) -> str:
    lower = filename.lower()

    if lower.endswith((".txt", ".csv")):
        return data.decode("utf-8", errors="ignore")

    if lower.endswith(".json"):
        try:
            loaded = json.loads(data.decode("utf-8", errors="ignore"))
            return json.dumps(loaded)
        except Exception:
            return data.decode("utf-8", errors="ignore")

    if lower.endswith(".pdf") and PdfReader:
        try:
            reader = PdfReader(io.BytesIO(data))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    return ""


def build_patient_from_dynamic_input(
    description: str,
    report_texts: List[str],
    manual_overrides: Dict,
) -> Tuple[PatientProfile, Dict]:
    joined_reports = "\n".join([x for x in report_texts if x.strip()])
    full_text = f"{description}\n{joined_reports}".strip()

    condition, matched_alias = extract_condition(full_text)
    if not condition:
        fallback = str(manual_overrides.get("condition", "")).strip().lower().replace(" ", "_")
        condition = fallback if fallback else ""

    age = manual_overrides.get("age")
    if age is None:
        age = extract_age(full_text) or 45

    allergies = manual_overrides.get("allergies") or extract_simple_list(ALLERGY_PAT, full_text)
    comorbidities = manual_overrides.get("comorbidities") or extract_comorbidities(full_text)
    current_medications = manual_overrides.get("current_medications") or extract_simple_list(MEDS_PAT, full_text)

    labs = extract_labs(full_text)
    pregnant = bool(manual_overrides.get("pregnant", False)) or ("pregnant" in _normalize(full_text))
    renal_impairment = bool(manual_overrides.get("renal_impairment", False)) or infer_renal_impairment(labs, full_text)
    liver_impairment = bool(manual_overrides.get("liver_impairment", False)) or infer_liver_impairment(labs, full_text)

    profile = PatientProfile(
        condition=condition,
        age=int(age),
        pregnant=pregnant,
        allergies=allergies,
        comorbidities=comorbidities,
        current_medications=current_medications,
        renal_impairment=renal_impairment,
        liver_impairment=liver_impairment,
    )

    trace = {
        "matched_condition_alias": matched_alias,
        "labs_detected": labs,
        "text_length": len(full_text),
        "reports_processed": len(report_texts),
    }
    return profile, trace
