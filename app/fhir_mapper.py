"""Maps extracted prescription data to a FHIR R4 MedicationRequest Bundle."""

import uuid

from app.models import ExtractionResult, Medication


def _build_dosage_text(med: Medication) -> str:
    """
    Construct a human-readable dosage instruction string from medication fields.

    Example: "500mg twice daily after meals for 30 days"
    """
    parts = []

    if med.dosage:
        parts.append(med.dosage)
    if med.frequency:
        parts.append(med.frequency)
    if med.instructions:
        parts.append(med.instructions)
    if med.duration:
        parts.append(f"for {med.duration}")

    return " ".join(parts) if parts else "As directed"


def _build_medication_request(
    med: Medication,
    patient_name: str,
    doctor_name: str,
    prescription_date: str | None,
) -> dict:
    """Create a single FHIR R4 MedicationRequest resource."""
    dosage_instruction = {"text": _build_dosage_text(med)}

    if med.route:
        dosage_instruction["route"] = {"coding": [{"display": med.route}]}

    resource = {
        "resourceType": "MedicationRequest",
        "id": str(uuid.uuid4()),
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": {"text": med.drug_name},
        "subject": {"display": patient_name},
        "requester": {"display": doctor_name},
        "dosageInstruction": [dosage_instruction],
    }

    if prescription_date:
        resource["authoredOn"] = prescription_date

    if med.purpose:
        resource["reasonCode"] = [{"text": med.purpose}]

    return resource


def _build_allergy_intolerance(allergy: str, patient_name: str) -> dict:
    """Create a single FHIR R4 AllergyIntolerance resource."""
    return {
        "resourceType": "AllergyIntolerance",
        "id": str(uuid.uuid4()),
        "clinicalStatus": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                    "code": "active",
                    "display": "Active",
                }
            ]
        },
        "patient": {"display": patient_name},
        "code": {"text": allergy},
    }


def build_fhir_bundle(extraction: ExtractionResult) -> dict:
    """
    Convert an ExtractionResult into a FHIR R4 Bundle of MedicationRequests
    and AllergyIntolerance resources.

    Args:
        extraction: Structured data from the LLM extraction step.

    Returns:
        A FHIR R4 Bundle dict with type "collection".
    """
    patient_name = extraction.patient_name or "Unknown Patient"
    doctor_name = extraction.doctor_name or "Unknown Practitioner"

    entries = []

    # Build MedicationRequest entries
    for med in extraction.medications:
        resource = _build_medication_request(
            med=med,
            patient_name=patient_name,
            doctor_name=doctor_name,
            prescription_date=extraction.prescription_date,
        )
        entries.append({"resource": resource})

    # Build AllergyIntolerance entries
    if extraction.allergies:
        for allergy in extraction.allergies:
            resource = _build_allergy_intolerance(
                allergy=allergy,
                patient_name=patient_name,
            )
            entries.append({"resource": resource})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }
