import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schemas import PatientProfile


class DrugRecommender:
    def __init__(self, knowledge_file: Optional[str] = None) -> None:
        api_dir = Path(__file__).resolve().parents[1]
        project_root = Path(__file__).resolve().parents[2]

        if knowledge_file:
            requested = Path(knowledge_file)
            if requested.is_absolute():
                candidates = [requested]
            else:
                candidates = [
                    Path.cwd() / requested,
                    project_root / requested,
                    api_dir / requested,
                ]
        else:
            candidates = [api_dir / "data" / "drug_knowledge.json"]

        knowledge_path = next((p for p in candidates if p.exists()), None)
        if knowledge_path is None:
            searched = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"Knowledge base not found. Searched: {searched}")

        with knowledge_path.open("r", encoding="utf-8") as f:
            self.db = json.load(f)

    def recommend(self, patient: PatientProfile) -> Dict:
        condition = patient.condition.strip().lower().replace(" ", "_")
        options = self.db.get("conditions", {}).get(condition, [])

        if not options:
            return {
                "condition": condition,
                "top_recommendations": [],
                "excluded_options": [],
                "clinical_review": (
                    "No curated guideline-backed options found for this condition in the local KB. "
                    "Add validated protocols before using for care decisions."
                ),
                "confidence": 0.15,
                "disclaimer": "Decision support only; requires clinician verification.",
            }

        ranked: List[Tuple[float, Dict]] = []
        excluded: List[Dict] = []

        risk_tags = set([c.lower() for c in patient.comorbidities])
        risk_tags.update([a.lower() for a in patient.allergies])
        if patient.pregnant:
            risk_tags.add("pregnancy")
        if patient.renal_impairment:
            risk_tags.add("chronic_kidney_disease")
            risk_tags.add("severe_renal_failure")
        if patient.liver_impairment:
            risk_tags.add("liver_disease")

        for opt in options:
            avoid_if = {s.lower() for s in opt.get("avoid_if", [])}
            caution_if = {s.lower() for s in opt.get("caution_if", [])}

            hard_hits = sorted(list(avoid_if.intersection(risk_tags)))
            if hard_hits:
                excluded.append(
                    {
                        "drug": opt["drug"],
                        "reason": f"Contraindicated due to: {', '.join(hard_hits)}",
                    }
                )
                continue

            score = 1.0
            if opt.get("first_line", False):
                score += 0.9
            caution_hits = sorted(list(caution_if.intersection(risk_tags)))
            score -= 0.25 * len(caution_hits)

            if any(med.lower() == opt["drug"].lower() for med in patient.current_medications):
                score -= 0.15

            ranked.append(
                (
                    max(score, 0.0),
                    {
                        "drug": opt["drug"],
                        "class": opt.get("class", "Unknown"),
                        "score": round(max(score, 0.0), 4),
                        "cautions": caution_hits,
                        "common_side_effects": opt.get("common_side_effects", []),
                        "dose_note": opt.get("dose_note", ""),
                    },
                )
            )

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = [r[1] for r in ranked[:5]]

        confidence = self._compute_confidence(total=len(options), excluded=len(excluded), top_count=len(top))
        clinical_review = self._build_review(patient=patient, top=top, excluded=excluded, confidence=confidence)

        return {
            "condition": condition,
            "top_recommendations": top,
            "excluded_options": excluded,
            "clinical_review": clinical_review,
            "confidence": confidence,
            "disclaimer": (
                "This tool provides evidence-informed triage support and is not a prescription engine. "
                "Final treatment selection must be made by a licensed clinician."
            ),
        }

    @staticmethod
    def _compute_confidence(total: int, excluded: int, top_count: int) -> float:
        if total == 0:
            return 0.1
        base = 0.55 + (0.25 * (top_count / total))
        penalty = 0.2 * (excluded / total)
        return round(max(min(base - penalty, 0.95), 0.1), 4)

    @staticmethod
    def _build_review(patient: PatientProfile, top: List[Dict], excluded: List[Dict], confidence: float) -> str:
        if not top:
            return (
                "All locally available options were excluded based on contraindications or missing evidence. "
                "Escalate to specialist review."
            )

        top_names = ", ".join([d["drug"] for d in top[:3]])
        exclusions = "; ".join([f"{e['drug']} ({e['reason']})" for e in excluded[:3]]) or "None"
        return (
            f"Patient-specific ranking generated from curated rules. Preferred options: {top_names}. "
            f"Excluded/penalized options: {exclusions}. "
            f"Model confidence: {confidence}. Validate renal/hepatic function, interactions, and local guidelines before prescribing."
        )
