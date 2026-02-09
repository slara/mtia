# MTIA - ML/AI Service

Central service for ML/AI tasks. Each capability lives in its own module under `modules/`.

## Structure

```
mtia/
├── api.py                          # FastAPI orchestrator (includes module routers)
├── modules/
│   └── stopreason/
│       ├── pipeline.py             # ML pipeline (extract, train, predict CLI)
│       ├── router.py               # FastAPI endpoints (POST /predict, GET /health)
│       ├── models/cic/             # Trained model artifacts
│       └── data/                   # Extracted training data
├── Dockerfile
└── pyproject.toml
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/stopreason/health` | Stop-reason module health |
| POST | `/stopreason/predict` | Predict top-k stop reasons |

### Predict request

```json
POST /stopreason/predict
{
  "dev_id": 1079,
  "duration": 60.0,
  "top_k": 3,
  "user_id": null,
  "timestamp": "2026-01-29T14:52:27"
}
```

## CLI (stopreason pipeline)

```bash
# Extract training data
docker compose exec mtia stopreason extract --client-id 31 --output /app/modules/stopreason/data/cic.parquet --verbose

# Train model
docker compose exec mtia stopreason train --data /app/modules/stopreason/data/cic.parquet --output /app/modules/stopreason/models/cic --verbose

# Predict from CLI
docker compose exec mtia stopreason predict --model-dir /app/modules/stopreason/models/cic --dev-id 1079 --verbose
```

## Adding a new module

1. Create `modules/newmodule/` with `__init__.py`, `pipeline.py`, `router.py`
2. In `api.py`, add: `from modules.newmodule import router` + `app.include_router(router)`
3. Add `"modules.newmodule"` to `packages` in `pyproject.toml`
