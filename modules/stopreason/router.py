"""FastAPI router for stop-reason predictions."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .pipeline import predict_stop_reasons

router = APIRouter(prefix="/stopreason")


class PredictRequest(BaseModel):
    dev_id: int
    duration: float = 60.0
    user_id: Optional[int] = None
    top_k: int = 3
    timestamp: Optional[str] = None


@router.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "module": "stopreason",
        "model_loaded": request.app.state.artifacts is not None,
        "n_classes": request.app.state.artifacts["metadata"]["n_classes"],
    }


@router.post("/predict")
def predict(req: PredictRequest, request: Request):
    try:
        result = predict_stop_reasons(
            engine=request.app.state.engine,
            artifacts=request.app.state.artifacts,
            dev_id=req.dev_id,
            duration=req.duration,
            user_id=req.user_id,
            top_k=req.top_k,
            timestamp=req.timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result
