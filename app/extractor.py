"""OpenAI Vision extraction pipeline for prescription images."""

import base64
import json
import logging

from openai import OpenAI, OpenAIError

from app.models import ExtractionResult

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    """Lazy-initialize the OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

EXTRACTION_PROMPT = """You are a medical data extraction assistant.
Extract all medications and patient information from this prescription image.
Return ONLY a JSON object in this exact format, no explanation:

{
  "patient_name": string or null,
  "doctor_name": string or null,
  "prescription_date": string or null,
  "prescription_number": string or null,
  "patient_age": string or null,
  "patient_dob": string or null,
  "patient_gender": string or null,
  "allergies": array of strings or null,
  "confidence": float between 0 and 1,
  "medications": [
    {
      "drug_name": string,
      "dosage": string or null,
      "frequency": string or null,
      "duration": string or null,
      "instructions": string or null,
      "route": string or null (e.g. "Oral", "Topical", "IV", "Intramuscular"),
      "purpose": string or null (e.g. "For fever", "Bacterial infection", "Diabetes")
    }
  ]
}

Rules:
- If no medications are found, return an empty medications array with confidence 0.0.
- Never invent or guess values. Use null if not clearly visible.
- Allergies must always be returned as an array, even if only one value (e.g. ["Seafood"]).
- Use null for allergies if no allergy information is present."""


class ExtractionError(Exception):
    """Raised when OpenAI extraction fails."""

    pass


def extract_prescription(image_bytes: bytes, content_type: str) -> ExtractionResult:
    """
    Send a prescription image to OpenAI Vision and extract structured data.

    Args:
        image_bytes: Raw image bytes.
        content_type: MIME type (image/jpeg or image/png).

    Returns:
        ExtractionResult with parsed medication data.

    Raises:
        ExtractionError: If the OpenAI API call fails or returns invalid data.
    """
    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{content_type};base64,{base64_image}"

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": EXTRACTION_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all medication data from this prescription image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "auto"},
                        },
                    ],
                },
            ],
            max_tokens=2000,
        )

        raw_content = response.choices[0].message.content
        logger.info("OpenAI raw response: %s", raw_content)

        parsed = json.loads(raw_content)
        return ExtractionResult(**parsed)

    except OpenAIError as e:
        logger.error("OpenAI API error: %s", str(e))
        raise ExtractionError(f"OpenAI API failed: {str(e)}") from e
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("Failed to parse OpenAI response: %s", str(e))
        raise ExtractionError(f"Failed to parse extraction result: {str(e)}") from e
    except Exception as e:
        logger.error("Unexpected extraction error: %s", str(e))
        raise ExtractionError(f"Unexpected error: {str(e)}") from e
