"""Tests for the /extract and /health endpoints."""

import io
import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.extractor import ExtractionError
from app.models import ExtractionResult, Medication

client = TestClient(app)


# ─── Health Check ────────────────────────────────────────────────────────────


def test_health():
    """GET /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ─── File Validation ─────────────────────────────────────────────────────────


def test_unsupported_file_type():
    """Uploading a non-image file returns 422."""
    fake_file = io.BytesIO(b"not an image")
    response = client.post(
        "/extract",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "Unsupported file type. Use JPG or PNG."


def test_file_too_large():
    """Uploading a file >2MB returns 422."""
    # Create a 6MB file
    large_file = io.BytesIO(b"\x00" * (6 * 1024 * 1024))
    response = client.post(
        "/extract",
        files={"file": ("big.jpg", large_file, "image/jpeg")},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "File size exceeds 2MB limit."


# ─── Successful Extraction ───────────────────────────────────────────────────

MOCK_EXTRACTION = ExtractionResult(
    patient_name="Rahul Sharma",
    doctor_name="Dr. Priya Nair",
    prescription_date="2025-03-15",
    confidence=0.91,
    allergies=["Seafood"],
    medications=[
        Medication(
            drug_name="Metformin",
            dosage="500mg",
            frequency="twice daily",
            duration="30 days",
            instructions="after meals",
            route="Oral",
            purpose="Diabetes",
        ),
        Medication(
            drug_name="Amlodipine",
            dosage="5mg",
            frequency="once daily",
            duration=None,
            instructions="in the morning",
        ),
    ],
)


@patch("app.main.extract_prescription")
def test_successful_extraction(mock_extract):
    """Valid image with mocked OpenAI returns FHIR Bundle."""
    mock_extract.return_value = MOCK_EXTRACTION

    # Minimal valid JPEG header
    fake_jpg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    response = client.post(
        "/extract",
        files={"file": ("prescription.jpg", fake_jpg, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    # Top-level fields
    assert data["confidence_score"] == 0.91
    assert data["fhir_bundle"]["resourceType"] == "Bundle"
    assert data["fhir_bundle"]["type"] == "collection"

    # Two MedicationRequest entries + 1 AllergyIntolerance
    entries = data["fhir_bundle"]["entry"]
    assert len(entries) == 3

    # First medication
    med1 = entries[0]["resource"]
    assert med1["resourceType"] == "MedicationRequest"
    assert med1["status"] == "active"
    assert med1["intent"] == "order"
    assert med1["medicationCodeableConcept"]["text"] == "Metformin"
    assert med1["subject"]["display"] == "Rahul Sharma"
    assert med1["requester"]["display"] == "Dr. Priya Nair"
    assert "500mg" in med1["dosageInstruction"][0]["text"]
    assert med1["authoredOn"] == "2025-03-15"

    # Second medication
    med2 = entries[1]["resource"]
    assert med2["medicationCodeableConcept"]["text"] == "Amlodipine"

    # Allergy entry
    allergy = entries[2]["resource"]
    assert allergy["resourceType"] == "AllergyIntolerance"
    assert allergy["code"]["text"] == "Seafood"


# ─── Route and Purpose in Bundle ─────────────────────────────────────────────


@patch("app.main.extract_prescription")
def test_route_and_purpose_in_bundle(mock_extract):
    """Route and purpose fields are mapped to FHIR dosageInstruction and reasonCode."""
    mock_extract.return_value = ExtractionResult(
        confidence=0.88,
        medications=[
            Medication(
                drug_name="Amoxicillin",
                dosage="500mg",
                frequency="Every 8 hours",
                route="Oral",
                purpose="Bacterial infection",
            ),
        ],
    )

    fake_jpg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    response = client.post(
        "/extract",
        files={"file": ("rx.jpg", fake_jpg, "image/jpeg")},
    )

    assert response.status_code == 200
    med = response.json()["fhir_bundle"]["entry"][0]["resource"]

    # Route mapped into dosageInstruction
    assert med["dosageInstruction"][0]["route"]["coding"][0]["display"] == "Oral"

    # Purpose mapped as reasonCode
    assert med["reasonCode"][0]["text"] == "Bacterial infection"


# ─── Allergy in Bundle ───────────────────────────────────────────────────────


@patch("app.main.extract_prescription")
def test_allergy_in_bundle(mock_extract):
    """Allergies are mapped as AllergyIntolerance entries in the FHIR bundle."""
    mock_extract.return_value = ExtractionResult(
        confidence=0.85,
        allergies=["Penicillin", "Shellfish"],
        medications=[
            Medication(drug_name="Ibuprofen", dosage="400mg"),
        ],
    )

    fake_jpg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    response = client.post(
        "/extract",
        files={"file": ("allergy.jpg", fake_jpg, "image/jpeg")},
    )

    assert response.status_code == 200
    entries = response.json()["fhir_bundle"]["entry"]

    # 1 MedicationRequest + 2 AllergyIntolerance
    assert len(entries) == 3

    allergy_entries = [
        e["resource"] for e in entries
        if e["resource"]["resourceType"] == "AllergyIntolerance"
    ]
    assert len(allergy_entries) == 2
    assert allergy_entries[0]["code"]["text"] == "Penicillin"
    assert allergy_entries[1]["code"]["text"] == "Shellfish"
    assert allergy_entries[0]["clinicalStatus"]["coding"][0]["code"] == "active"
    assert allergy_entries[0]["patient"]["display"] == "Unknown Patient"


# ─── No Medications Found ────────────────────────────────────────────────────


@patch("app.main.extract_prescription")
def test_no_medications_found(mock_extract):
    """Empty extraction returns Bundle with empty entries and confidence 0.0."""
    mock_extract.return_value = ExtractionResult(
        confidence=0.0,
        medications=[],
    )

    fake_png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    response = client.post(
        "/extract",
        files={"file": ("empty.png", fake_png, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["confidence_score"] == 0.0
    assert data["fhir_bundle"]["entry"] == []


# ─── OpenAI Failure ──────────────────────────────────────────────────────────


@patch("app.main.extract_prescription")
def test_openai_failure(mock_extract):
    """OpenAI API failure returns 502."""
    mock_extract.side_effect = ExtractionError("Connection timeout")

    fake_jpg = io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    response = client.post(
        "/extract",
        files={"file": ("fail.jpg", fake_jpg, "image/jpeg")},
    )

    assert response.status_code == 502
    assert response.json()["error"] == "Extraction service unavailable."
