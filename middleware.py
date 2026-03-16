"""
watermark/middleware.py
-----------------------
Drop-in integration for the Aegis proxy (server.js calls this via FastAPI).

Adds two endpoints to the existing scorer FastAPI app:

  POST /watermark/embed    — call after every LLM response
  POST /watermark/detect   — call to investigate a suspect clone

HOW TO PLUG INTO AEGIS
-----------------------
In src/scorer/scorer.py, add at the bottom:

    from watermark.middleware import add_watermark_routes
    add_watermark_routes(app)

Then in src/proxy/server.js, after receiving the LLM response:

    const wm = await fetch('http://localhost:8000/watermark/embed', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: llmResponse, session_id: sessionId })
    }).then(r => r.json());

    // Return wm.watermarked_text to the caller instead of the raw response
    return wm.watermarked_text;
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .key import WatermarkKey, KeyStore
from .embedder import WatermarkEmbedder
from .detector import WatermarkDetector


# ── Request / Response models ────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str
    session_id: Optional[str] = None   # logged for audit trail


class EmbedResponse(BaseModel):
    watermarked_text: str
    substitutions_made: int
    eligible_words: int
    key_version: int
    session_id: Optional[str]


class DetectRequest(BaseModel):
    text: str
    label: Optional[str] = None        # e.g. "suspect_clone_v1"


class DetectResponse(BaseModel):
    watermark_detected: bool
    confidence: str
    z_score: float
    p_value: float
    green_fraction: float
    total_eligible: int
    key_version_matched: Optional[int]
    summary: str
    report: str                         # Full forensic report text


# ── Singleton setup ──────────────────────────────────────────────────────────

def _build_keystore() -> KeyStore:
    """
    Build the keystore from environment variables.
    Falls back to a generated key for development (NOT for production).
    """
    store = KeyStore.from_env()
    if not store.all_versions():
        # Dev fallback — log a loud warning
        import warnings
        warnings.warn(
            "No AEGIS_WATERMARK_KEY set. Generating a temporary key. "
            "This key will NOT persist across restarts. Set AEGIS_WATERMARK_KEY in production.",
            RuntimeWarning,
            stacklevel=2,
        )
        store.add(WatermarkKey.generate(version=1))
    return store


_keystore: Optional[KeyStore] = None
_embedder: Optional[WatermarkEmbedder] = None
_detector: Optional[WatermarkDetector] = None


def _get_components():
    global _keystore, _embedder, _detector
    if _keystore is None:
        _keystore = _build_keystore()
        _embedder = WatermarkEmbedder(_keystore.latest())
        _detector = WatermarkDetector.from_keystore(_keystore)
    return _embedder, _detector


# ── Route registration ───────────────────────────────────────────────────────

def add_watermark_routes(app: FastAPI):
    """
    Register watermark endpoints onto an existing FastAPI app.

    Usage in scorer.py:
        from watermark.middleware import add_watermark_routes
        add_watermark_routes(app)
    """

    @app.post("/watermark/embed", response_model=EmbedResponse)
    async def embed(req: EmbedRequest):
        """
        Embed a watermark into an LLM response.
        Call this for every response before returning it to the API caller.
        """
        embedder, _ = _get_components()

        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")

        result = embedder.embed(req.text)

        return EmbedResponse(
            watermarked_text=result.watermarked_text,
            substitutions_made=result.substitutions_made,
            eligible_words=result.eligible_words,
            key_version=result.key_version,
            session_id=req.session_id,
        )

    @app.post("/watermark/detect", response_model=DetectResponse)
    async def detect(req: DetectRequest):
        """
        Detect whether a text carries the Aegis watermark.
        Use this to investigate suspected clone model outputs.
        """
        _, detector = _get_components()

        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="text must not be empty")

        result = detector.detect(req.text)

        return DetectResponse(
            watermark_detected=result.watermark_detected,
            confidence=result.confidence,
            z_score=round(result.z_score, 4),
            p_value=round(result.p_value, 8),
            green_fraction=round(result.green_fraction, 4),
            total_eligible=result.total_eligible,
            key_version_matched=result.key_version_matched,
            summary=result.summary,
            report=result.to_report(),
        )

    @app.get("/watermark/health")
    async def health():
        """Check the watermark service is running and keys are loaded."""
        embedder, _ = _get_components()
        return {
            "status": "ok",
            "key_versions": _keystore.all_versions(),
            "active_key_version": _keystore.latest().version,
        }
