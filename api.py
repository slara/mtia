"""MTIA — central ML/AI service."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import create_engine

from modules.stopreason import load_model_artifacts, router as stopreason_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.environ["MODEL_DIR"]
    database_url = os.environ["DATABASE_URL"]

    app.state.artifacts = load_model_artifacts(model_dir)
    app.state.engine = create_engine(database_url)

    n_classes = app.state.artifacts["metadata"]["n_classes"]
    print(f"Model loaded: {n_classes} classes from {model_dir}")
    yield


app = FastAPI(title="MTIA - ML/AI Service", lifespan=lifespan)
app.include_router(stopreason_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "mtia"}
