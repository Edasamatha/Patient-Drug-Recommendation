from typing import List, Optional
from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    condition: str = Field(..., description="Primary diagnosed condition")
    age: int = Field(..., ge=0, le=120)
    pregnant: bool = False
    allergies: List[str] = Field(default_factory=list)
    comorbidities: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    renal_impairment: bool = False
    liver_impairment: bool = False


class SentimentRequest(BaseModel):
    reviews: List[str] = Field(..., min_length=1, max_length=100)
    model: Optional[str] = None


class RecommendResponse(BaseModel):
    condition: str
    top_recommendations: List[dict]
    excluded_options: List[dict]
    clinical_review: str
    confidence: float
    disclaimer: str


class SentimentResponse(BaseModel):
    review_count: int
    per_review: List[dict]
    aggregate: dict
