# MTIA - Servicio ML/AI

Servicio central para tareas de ML/AI. Cada funcionalidad vive en su propio módulo dentro de `modules/`.

## Estructura

```
mtia/
├── api.py                          # Orquestador FastAPI (incluye routers de módulos)
├── modules/
│   └── stopreason/
│       ├── pipeline.py             # Pipeline ML (extract, train, predict CLI)
│       ├── router.py               # Endpoints FastAPI (POST /predict, GET /health)
│       ├── models/cic/             # Artefactos del modelo entrenado
│       └── data/                   # Datos de entrenamiento extraídos
├── Dockerfile
└── pyproject.toml
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Chequeo de salud del servicio |
| GET | `/stopreason/health` | Salud del módulo stop-reason |
| POST | `/stopreason/predict` | Predecir top-k razones de detención |

### Request de predicción

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

## CLI (pipeline stopreason)

Cuando una máquina se detiene en la línea de producción, el sistema predice la razón de detención más probable usando un clasificador LightGBM entrenado con datos históricos de detenciones. El pipeline tiene tres pasos: **extract** → **train** → **predict**.

### Extract

Extrae eventos históricos de detención desde PostgreSQL (`j_s_reg`, `j_device`, `j_cod_state`, `j_line`) y construye un vector de features por cada evento de detención:

- **Temporales:** hora, día de la semana, mes
- **Contexto del dispositivo:** ID del dispositivo, operador, duración de la detención
- **Historial:** 2 razones de detención anteriores, tiempo desde la última detención
- **Topología de línea:** posición en la línea, largo de la línea, y estado de 4 vecinos (2 aguas arriba, 2 aguas abajo) en una ventana de 5 minutos

Las razones de detención con menos de `--min-samples` ocurrencias (por defecto 10) se filtran. Genera un archivo Parquet.

```bash
docker compose exec mtia stopreason extract --client-id 31 --output /app/modules/stopreason/data/cic.parquet --verbose
```

### Train

Entrena un clasificador LightGBM multi-clase. Compara dos modelos: solo-dispositivo (9 features) vs dispositivo+línea (25 features con contexto de vecinos), y persiste el modelo dispositivo+línea (mejor accuracy top-3).

Artefactos guardados: `model.joblib`, `label_encoder.joblib`, `metadata.json`.

Resultados CIC: ~85% top-1, ~95% top-3, ~97% top-5 (69 clases).

```bash
docker compose exec mtia stopreason train --data /app/modules/stopreason/data/cic.parquet --output /app/modules/stopreason/models/cic --verbose
```

### Predict

Consulta la base de datos en tiempo real para obtener contexto (info del dispositivo, detenciones recientes, estado de vecinos), arma el mismo vector de 25 features usado durante el entrenamiento, y retorna las top-k predicciones con scores de confianza y descripciones de razones de detención.

Disponible como comando CLI y como endpoint `POST /stopreason/predict`.

```bash
docker compose exec mtia stopreason predict --model-dir /app/modules/stopreason/models/cic --dev-id 1079 --verbose
```

## Agregar un nuevo módulo

1. Crear `modules/newmodule/` con `__init__.py`, `pipeline.py`, `router.py`
2. En `api.py`, agregar: `from modules.newmodule import router` + `app.include_router(router)`
3. Agregar `"modules.newmodule"` a `packages` en `pyproject.toml`
