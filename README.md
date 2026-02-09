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

When a machine stops on the production line, the system predicts the most likely stop reason using a LightGBM classifier trained on historical stop data. The pipeline has three steps: **extract** → **train** → **predict**.

### Extract

Pulls historical stop events from PostgreSQL (`j_s_reg`, `j_device`, `j_cod_state`, `j_line`) and builds a feature vector per stop event:

- **Temporal:** hour, day of week, month
- **Device context:** device ID, operator, stop duration
- **History:** previous 2 stop reasons, time since last stop
- **Line topology:** position in line, line length, and status of 4 neighbors (2 upstream, 2 downstream) within a 5-minute window

Stop reasons with fewer than `--min-samples` occurrences (default 10) are filtered out. Outputs a Parquet file.

```bash
docker compose exec mtia stopreason extract --client-id 31 --output /app/modules/stopreason/data/cic.parquet --verbose
```

### Train

Trains a LightGBM multi-class classifier. Compares two models: device-only (9 features) vs device+line (25 features with neighbor context), and persists the device+line model (better top-3 accuracy).

Saved artifacts: `model.joblib`, `label_encoder.joblib`, `metadata.json`.

CIC results: ~85% top-1, ~95% top-3, ~97% top-5 (69 classes).

```bash
docker compose exec mtia stopreason train --data /app/modules/stopreason/data/cic.parquet --output /app/modules/stopreason/models/cic --verbose
```

### Predict

Queries the live database for real-time context (device info, recent stops, neighbor status), assembles the same 25-feature vector used during training, and returns top-k predictions with confidence scores and stop reason descriptions.

Available as both CLI command and `POST /stopreason/predict` endpoint.

```bash
docker compose exec mtia stopreason predict --model-dir /app/modules/stopreason/models/cic --dev-id 1079 --verbose
```

## Adding a new module

1. Create `modules/newmodule/` with `__init__.py`, `pipeline.py`, `router.py`
2. In `api.py`, add: `from modules.newmodule import router` + `app.include_router(router)`
3. Add `"modules.newmodule"` to `packages` in `pyproject.toml`
