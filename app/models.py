"""Pydantic models for request/response contracts."""

from pydantic import BaseModel, Field
from typing import Optional


class Medication(BaseModel):
    """A single medication extracted from the prescription."""

    drug_name: str
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    instructions: Optional[str] = None
    route: Optional[str] = None
    purpose: Optional[str] = None


class ExtractionResult(BaseModel):
    """Full structured extraction from OpenAI Vision."""

    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    prescription_date: Optional[str] = None
    prescription_number: Optional[str] = None
    patient_age: Optional[str] = None
    patient_dob: Optional[str] = None
    patient_gender: Optional[str] = None
    allergies: Optional[list[str]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    medications: list[Medication] = []


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str


class ExtractResponse(BaseModel):
    """Successful extraction response with FHIR bundle."""

    confidence_score: float = Field(ge=0.0, le=1.0)
    fhir_bundle: dict
