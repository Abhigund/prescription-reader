"""Prescription Extraction API — FastAPI application."""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.extractor import ExtractionError, extract_prescription
from app.fhir_mapper import build_fhir_bundle
from app.models import ExtractResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 2 * 1024 * 1024  # 2MB

app = FastAPI(
    title="Prescription Extractor API",
    description=(
        "Extracts structured medication data from printed prescription images "
        "using OpenAI Vision and returns FHIR R4 MedicationRequest Bundles."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Health check endpoint for Render uptime monitoring."""
    return {"status": "ok"}


@app.post(
    "/extract",
    response_model=ExtractResponse,
    responses={
        422: {"description": "Invalid file type or size"},
        502: {"description": "Extraction service unavailable"},
    },
)
async def extract_prescription_endpoint(file: UploadFile = File(...)):
    """
    Accept a prescription image and return a FHIR R4 MedicationRequest Bundle.

    - **file**: Prescription image (JPG or PNG, max 5MB)
    """
    # --- Validate file type ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning("Rejected file type: %s", file.content_type)
        return JSONResponse(
            status_code=422,
            content={"error": "Unsupported file type. Use JPG or PNG."},
        )

    # --- Read and validate file size ---
    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        logger.warning("Rejected file size: %d bytes", len(image_bytes))
        return JSONResponse(
            status_code=422,
            content={"error": "File size exceeds 2MB limit."},
        )

    # --- Extract via OpenAI Vision ---
    try:
        extraction = extract_prescription(image_bytes, file.content_type)
    except ExtractionError as e:
        logger.error("Extraction failed: %s", str(e))
        return JSONResponse(
            status_code=502,
            content={"error": "Extraction service unavailable."},
        )

    # --- Map to FHIR Bundle ---
    fhir_bundle = build_fhir_bundle(extraction)

    logger.info(
        "Extraction complete — %d medications, confidence: %.2f",
        len(extraction.medications),
        extraction.confidence,
    )

    return ExtractResponse(
        confidence_score=extraction.confidence,
        fhir_bundle=fhir_bundle,
    )
