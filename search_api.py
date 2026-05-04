"""
Person Search API
=================
FastAPI service that accepts a natural-language text query, encodes it with
CLIP, and returns the top-K matching person crops from Qdrant.  Includes a
snapshot proxy to the Smart NVR so the frontend can fetch camera frames
without CORS issues.

Run:
    uvicorn search_api:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
# dummy imports
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import FieldCondition, Filter, MatchValue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("search_api")

_state: dict = {}

# ``sensor_id`` from Kafka/Qdrant often looks like ``office_cam_01`` while Smart NVR
# APIs expect slugs such as ``office-cam01``. Map explicit entries in config first.
_STREAMER_CAM_RE = re.compile(r"^(.+)_cam_0*(\d+)$", re.IGNORECASE)


def _infer_nvr_camera_from_streamer_style(raw: str) -> Optional[str]:
    m = _STREAMER_CAM_RE.match((raw or "").strip())
    if not m:
        return None
    prefix, num_s = m.group(1), m.group(2)
    prefix_dashed = prefix.replace("_", "-")
    n = int(num_s, 10)
    suffix = f"{n:02d}" if n < 100 else str(n)
    return f"{prefix_dashed}-cam{suffix}"


def _snapshot_nvr_camera_name(*, sensor_id: Optional[str], camera: str) -> str:
    """Resolve the ``camera`` query parameter the NVR expects."""
    sm = _state.get("sensor_camera_map") or {}
    for key in (sensor_id, camera):
        if key and key in sm:
            return sm[key]
    for raw in (sensor_id, camera):
        if not raw:
            continue
        inferred = _infer_nvr_camera_from_streamer_style(raw)
        if inferred:
            return inferred
    return camera


def _qdrant_client_from_config(qdrant_cfg: dict) -> QdrantClient:
    # qdrant-client defaults https=True whenever api_key is set; local Qdrant is HTTP on 6333.
    kwargs: dict = {
        "host": qdrant_cfg.get("host", "localhost"),
        "port": int(qdrant_cfg.get("port", 6333)),
        "https": bool(qdrant_cfg.get("https", False)),
        "check_compatibility": bool(qdrant_cfg.get("check_compatibility", False)),
    }
    api_key = qdrant_cfg.get("api_key") or os.environ.get("QDRANT_API_KEY")
    if api_key:
        kwargs["api_key"] = str(api_key).strip()
    return QdrantClient(**kwargs)


def _qdrant_error_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, UnexpectedResponse) and exc.status_code == 401:
        return HTTPException(
            status_code=503,
            detail=(
                "Qdrant returned 401: this Qdrant instance is running with API-key auth enabled. "
                "This app does not need an OpenAI key. Either (a) put your Qdrant key in "
                "qdrant.api_key / env QDRANT_API_KEY, or (b) turn off Qdrant auth: do not set "
                "QDRANT__SERVICE__API_KEY when starting Qdrant; if auth was enabled before, you may "
                "need a fresh storage volume or remove the key from Qdrant's stored config."
            ),
        )
    msg = str(exc)
    if "WRONG_VERSION_NUMBER" in msg or "wrong version number" in msg.lower():
        return HTTPException(
            status_code=503,
            detail=(
                "TLS error talking to Qdrant (often HTTPS client vs HTTP server). "
                'For local Qdrant use "https": false in qdrant config (default). '
                "For Qdrant Cloud set \"https\": true."
            ),
        )
    return HTTPException(status_code=503, detail=msg)

_REPO_ROOT = Path(__file__).resolve().parent


def _resolve_config_path() -> Path:
    """Prefer ./config.json next to this file; never open a mistaken config.json directory."""
    preferred = _REPO_ROOT / "config.json"
    example = _REPO_ROOT / "config.example.json"
    if preferred.is_file():
        return preferred
    if preferred.is_dir():
        logger.warning(
            "Ignoring %s (it is a directory, not a JSON file). Remove it and copy "
            "config.example.json to config.json. Using config.example.json for now.",
            preferred,
        )
    if example.is_file():
        return example
    raise FileNotFoundError(
        f"No readable config: need {preferred} (file) or {example}"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = _resolve_config_path()
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    device = "cpu"
    model = None
    tokenizer = None

    qdrant_cfg = config["qdrant"]
    qdrant = _qdrant_client_from_config(qdrant_cfg)
    collection_name = qdrant_cfg.get("collection_name", "persons")

    query_vector_dim = 512
    try:
        info = qdrant.get_collection(collection_name)
        query_vector_dim = int(info.config.params.vectors.size)
        logger.info(
            "Collection %r vector size=%d (dummy text embeddings will match this dim)",
            collection_name,
            query_vector_dim,
        )
    except Exception as e:
        logger.warning(
            "Could not read collection %r (%s); using query vector dim %d until it exists.",
            collection_name,
            e,
            query_vector_dim,
        )

    _state["model"]     = model
    _state["tokenizer"] = tokenizer
    _state["device"]    = device
    _state["qdrant"] = qdrant
    _state["collection_name"] = collection_name
    _state["query_vector_dim"] = query_vector_dim
    _state["config"] = config
    _state["sensor_camera_map"] = config.get("sensor_camera_map", {})
    _state["nvr_base_url"] = config.get("nvr_base_url", "http://localhost:8009")
    _state["http_client"] = httpx.AsyncClient(timeout=15.0)

    logger.info("Search API ready")
    yield

    await _state["http_client"].aclose()
    _state.clear()


app = FastAPI(title="Person Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# CLIP text encoding
# ---------------------------------------------------------------------------

def _encode_text(text: str) -> List[float]:
    """Placeholder embedding; dimension matches the Qdrant collection (see lifespan)."""
    import numpy as np

    dim = int(_state.get("query_vector_dim", 512))
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.tolist()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query:           str
    top_k:           int   = 10
    score_threshold: float = 0.05
    sensor_id:  Optional[str] = None
    pad_index:  Optional[int] = None


class PersonResult(BaseModel):
    id:           str
    score:        float
    sensor_id:    str
    camera_name:  str
    tracker_id:   int
    timestamp:    str
    bbox:         List[float]
    frame_number: int
    confidence:   float
    pad_index:    int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/search", response_model=List[PersonResult])
def search(req: SearchRequest):
    """Search for persons matching a natural-language description."""
    query_vector = _encode_text(req.query)

    must = []
    if req.sensor_id:
        must.append(FieldCondition(key="sensor_id", match=MatchValue(value=req.sensor_id)))
    if req.pad_index is not None:
        must.append(FieldCondition(key="pad_index", match=MatchValue(value=req.pad_index)))

    qdrant_filter = Filter(must=must) if must else None

    collection = _state.get("collection_name", "persons")
    try:
        response = _state["qdrant"].query_points(
            collection_name=collection,
            query=query_vector,
            limit=req.top_k,
            query_filter=qdrant_filter,
            score_threshold=req.score_threshold,
        )
        hits = response.points
    except UnexpectedResponse as e:
        raise _qdrant_error_http_exception(e) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    sensor_map = _state["sensor_camera_map"]
    results = []
    for hit in hits:
        p = hit.payload
        sid = p.get("sensor_id", "")
        results.append(PersonResult(
            id=str(hit.id),
            score=round(hit.score, 4),
            sensor_id=sid,
            camera_name=sensor_map.get(sid, sid),
            tracker_id=p.get("tracker_id", -1),
            timestamp=p.get("timestamp", ""),
            bbox=p.get("bbox", []),
            frame_number=p.get("frame_number", 0),
            confidence=p.get("confidence", 0.0),
            pad_index=p.get("pad_index", 0),
        ))

    return results


@app.get("/snapshot")
async def snapshot_proxy(
    camera: str = Query(..., description="UI label or raw sensor id; mapped to NVR id when possible"),
    timestamp: str = Query(..., description="ISO 8601 timestamp"),
    quality: int = Query(85, ge=1, le=95),
    sensor_id: Optional[str] = Query(
        None,
        description="Qdrant/Kafka sensor_id; used with sensor_camera_map to pick the NVR camera",
    ),
):
    """
    Proxy to Smart NVR /snapshot endpoint.  The frontend calls this to avoid
    CORS issues.  The NVR auto-cleans the JPEG file after serving.
    """
    nvr = _state["nvr_base_url"]
    url = f"{nvr}/snapshot"
    sid = (sensor_id or "").strip() or None
    nvr_camera = _snapshot_nvr_camera_name(sensor_id=sid, camera=camera)
    if nvr_camera != camera or (sid and nvr_camera != sid):
        logger.info("Snapshot proxy camera %r (sensor_id=%r) -> NVR %r", camera, sid, nvr_camera)
    params = {"camera": nvr_camera, "timestamp": timestamp, "quality": quality}

    try:
        resp = await _state["http_client"].get(url, params=params)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"NVR unreachable: {e}")

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.text[:500],
        )

    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
    )


@app.get("/health")
def health():
    return {"status": "ok", "device": _state.get("device", "unknown")}


@app.get("/stats")
def stats():
    collection = _state.get("collection_name", "persons")
    try:
        info = _state["qdrant"].get_collection(collection)
        return {
            "vectors_count":         info.points_count,
            "status":                str(info.status),
        }
    except UnexpectedResponse as e:
        raise _qdrant_error_http_exception(e) from e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Static files — serve the frontend UI
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root_redirect():
    return RedirectResponse(url="/ui/")


@app.get("/ui", include_in_schema=False)
def ui_redirect_trailing_slash():
    return RedirectResponse(url="/ui/")


ui_path = _REPO_ROOT / "ui"
if ui_path.is_dir():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")
else:
    logger.warning("UI directory missing at %s — /ui will not be served", ui_path)
