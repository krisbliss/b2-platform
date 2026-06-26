"""POC-1 FastAPI endpoint — POST /score.

Accepts multipart image + form fields, constructs a Submission, runs the
pipeline, and returns a ReliabilityResult as JSON.

Run locally:
    uvicorn poc.api:app --reload
"""

from __future__ import annotations

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from tools.death_certificate_pipeline.models import ReliabilityResult, Submission
from tools.death_certificate_pipeline.pipeline import run_pipeline

app = FastAPI(
    title="B2 POC-1 — Reliability Scorer",
    description="Scores the reliability of a death certificate submission.",
    version="0.1.0",
)


@app.post("/score", response_model=ReliabilityResult)
async def score(
    image: UploadFile = File(..., description="Certificate image (PNG/JPEG/WEBP)."),
    narrative: str = Form(..., description="Free-text narrative from the claimant."),
    case_fields: str = Form(
        default="{}",
        description="JSON object of structured case metadata.",
    ),
) -> ReliabilityResult:
    """Score the reliability of a submitted death certificate.

    Returns a ReliabilityResult with score (1–100), band, sub-scores,
    flags, justification, and extracted certificate fields.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="image must not be empty")

    try:
        fields = json.loads(case_fields)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"case_fields is not valid JSON: {exc}",
        ) from exc

    try:
        submission = Submission(
            image=image_bytes,
            narrative=narrative,
            case_fields=fields,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return await run_pipeline(submission)
