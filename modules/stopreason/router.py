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
    client_models = request.app.state.client_models
    return {
        "status": "ok",
        "module": "stopreason",
        "clients_loaded": sorted(client_models.keys()),
        "n_clients": len(client_models),
    }


@router.post("/predict")
def predict(req: PredictRequest, request: Request):
    try:
        result = predict_stop_reasons(
            engine=request.app.state.engine,
            client_models=request.app.state.client_models,
            dev_id=req.dev_id,
            duration=req.duration,
            user_id=req.user_id,
            top_k=req.top_k,
            timestamp=req.timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result
