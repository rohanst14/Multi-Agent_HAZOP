# Implementation notes

## Components
- Backend: local threaded HTTP server and asynchronous study jobs.
- Prompt library: injected helper functions prevent circular imports.
- P&ID parser: local OpenCV and Tesseract extraction, optional vision callback, merged structured output. The standalone parser is included separately; the supplied backend does not directly import it.

## HTTP interface
- GET /api/health
- GET /api/models
- POST /api/hazop
- GET /api/jobs/{id}
- GET /api/jobs/{id}/status
- POST /api/jobs/{id}/stop
- GET /api/jobs/{id}/report.docx
- GET /api/jobs/{id}/worksheet.xlsx

The original browser frontend (src/static/) is required to launch the backend as provided. Do not commit API keys, confidential P&IDs or generated study records. External model calls may transmit supplied study data to the configured provider.

## Validation
All three supplied modules passed Python AST parsing and both backend and parser imported successfully in the local environment after filename normalization. Git blob hashes match the uploaded source text. Live provider requests, frontend startup and a full HAZOP study were not exercised. The original report remains unchanged.
