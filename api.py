"""FastAPI prediction service for ml-stopreason."""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine

from stopreason import load_model_artifacts, predict_stop_reasons


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.environ["MODEL_DIR"]
    database_url = os.environ["DATABASE_URL"]

    app.state.artifacts = load_model_artifacts(model_dir)
    app.state.engine = create_engine(database_url)

    n_classes = app.state.artifacts["metadata"]["n_classes"]
    print(f"Model loaded: {n_classes} classes from {model_dir}")
    yield


app = FastAPI(title="Stop Reason Predictor", lifespan=lifespan)


class PredictRequest(BaseModel):
    dev_id: int
    duration: float = 60.0
    user_id: Optional[int] = None
    top_k: int = 3
    timestamp: Optional[str] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.artifacts is not None,
        "n_classes": app.state.artifacts["metadata"]["n_classes"],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        result = predict_stop_reasons(
            engine=app.state.engine,
            artifacts=app.state.artifacts,
            dev_id=req.dev_id,
            duration=req.duration,
            user_id=req.user_id,
            top_k=req.top_k,
            timestamp=req.timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return result
