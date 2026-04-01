# 💊 Prescription Extractor API

A FastAPI service that accepts printed prescription images, extracts structured medication data using **OpenAI Vision API** (`gpt-4o`), and returns a **FHIR R4-compliant MedicationRequest Bundle** as JSON.

## ✨ Features

- **Image-to-Data**: Upload a prescription image → get structured medication data
- **FHIR R4 Compliant**: Output follows the HL7 FHIR R4 MedicationRequest specification
- **Confidence Scoring**: Each extraction includes a confidence score (0.0–1.0)
- **Zero-Hallucination Prompting**: LLM returns `null` for uncertain fields instead of guessing
- **Production Error Handling**: Proper HTTP status codes for all failure scenarios

## 🏗 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Framework | FastAPI |
| AI / Extraction | OpenAI API (`gpt-4o`, Vision) |
| Output Standard | FHIR R4 (MedicationRequest) |
| Deployment | Render (auto-deploy from GitHub) |

## 📁 Project Structure

```
prescription-extractor/
├── app/
│   ├── main.py              # FastAPI app, route definitions
│   ├── extractor.py         # OpenAI Vision call + prompt
│   ├── fhir_mapper.py       # Maps extracted JSON → FHIR Bundle
│   └── models.py            # Pydantic request/response models
├── tests/
│   └── test_extract.py      # Endpoint tests (mocked OpenAI)
├── prescription/
│   └── image.png            # Sample prescription image
├── sample/
│   └── sample_output.json   # Expected FHIR output
├── .env.example
├── requirements.txt
├── render.yaml
└── README.md
```

## 🚀 Local Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/prescription-extractor.git
cd prescription-extractor

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

The API is now live at `http://localhost:8000`

- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

## 📡 API Endpoints

### `GET /health`

Health check for uptime monitoring.

```bash
curl http://localhost:8000/health
```

```json
{ "status": "ok" }
```

### `POST /extract`

Upload a prescription image and receive a FHIR R4 MedicationRequest Bundle.

```bash
curl -X POST \
  -F "file=@prescription/image.png" \
  http://localhost:8000/extract
```

**Success Response (200):**

```json
{
  "confidence_score": 0.91,
  "fhir_bundle": {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
      {
        "resource": {
          "resourceType": "MedicationRequest",
          "status": "active",
          "intent": "order",
          "medicationCodeableConcept": {
            "text": "Metformin"
          },
          "subject": {
            "display": "Rahul Sharma"
          },
          "requester": {
            "display": "Dr. Priya Nair"
          },
          "dosageInstruction": [
            {
              "text": "500mg twice daily after meals for 30 days"
            }
          ]
        }
      }
    ]
  }
}
```

**Error Responses:**

| Scenario | Status | Response |
|----------|--------|----------|
| Unsupported file type | 422 | `{ "error": "Unsupported file type. Use JPG or PNG." }` |
| File exceeds 5MB | 422 | `{ "error": "File size exceeds 5MB limit." }` |
| OpenAI API failure | 502 | `{ "error": "Extraction service unavailable." }` |

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Tests use mocked OpenAI responses — no API key required.

## ☁️ Deployment (Render)

1. Push this repo to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/) → **New +** → **Blueprint**
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Set `OPENAI_API_KEY` in the Render environment variables
5. Deploy! Your API will be live at `https://prescription-reader.onrender.com`

## 📋 FHIR R4 Compliance

The API outputs a FHIR R4 `Bundle` of type `collection` containing `MedicationRequest` resources. Each resource includes:

- `medicationCodeableConcept` — Drug name as display text
- `subject` — Patient reference (extracted or placeholder)
- `requester` — Practitioner reference (extracted or placeholder)
- `dosageInstruction` — Human-readable dosage, frequency, duration, and instructions
- `authoredOn` — Prescription date (when available)

## 📄 License

MIT
