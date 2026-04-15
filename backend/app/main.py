from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .config import SETTINGS
from .schemas import CommitRequest, CommitResponse, PreviewRequest, PreviewResponse, SessionCreated
from .sessions import STORE


app = FastAPI(title="spark_sam_viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if SETTINGS.cors_origin == "*" else [SETTINGS.cors_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/sessions", response_model=SessionCreated)
async def create_session(file: UploadFile = File(...)) -> SessionCreated:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".ply":
        raise HTTPException(status_code=400, detail="Only .ply uploads are supported in v1.")

    try:
        payload = await file.read()
        state = STORE.create_session(file.filename or "upload.ply", payload)
        return SessionCreated(
            sessionId=state.session_id,
            numSplats=int(state.gaussians.get_xyz.shape[0]),
            bbox=STORE.build_bbox(state),
            warnings=[],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/preview", response_model=PreviewResponse)
def preview(session_id: str, payload: PreviewRequest) -> PreviewResponse:
    try:
        return PreviewResponse(**STORE.preview(session_id, payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/commit", response_model=CommitResponse)
def commit(session_id: str, payload: CommitRequest) -> CommitResponse:
    try:
        return CommitResponse(**STORE.commit(session_id, payload.op))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/export/mask")
def export_mask(session_id: str) -> Response:
    try:
        return Response(
            STORE.export_mask_bytes(session_id),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="mask.npy"'},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc


@app.get("/api/sessions/{session_id}/export/ply")
def export_ply(session_id: str) -> Response:
    try:
        return Response(
            STORE.export_ply_bytes(session_id),
            media_type="application/octet-stream",
            headers={"Content-Disposition": 'attachment; filename="object_3dgs.ply"'},
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Unknown session.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
