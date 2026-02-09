#!/usr/bin/env python3
"""Stop Reason Suggester — ML pipeline for predicting machine stop reasons.

Commands:
    extract  Pull historical stop data from the database into a Parquet file.
    train    Train a LightGBM model on extracted data.
    predict  Predict top-k stop reasons for a given device.
    serve    Start the FastAPI prediction server.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def get_engine(database_url=None):
    url = database_url or os.environ.get("DATABASE_URL")
    if not url:
        click.echo("Error: DATABASE_URL not set. Pass --database-url or set the env var.", err=True)
        sys.exit(1)
    return create_engine(url)


# ---------------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------------

EXTRACT_SQL = text("""
WITH line_topo AS (
    SELECT d.id AS dev_id, d.line_id, d.index AS position_in_line,
           COUNT(*) OVER (PARTITION BY d.line_id) AS line_length
    FROM j_device d
    WHERE d.client_id = :client_id
      AND d.visible = 1
      AND d.line_id IS NOT NULL
      AND d.index IS NOT NULL
),
neighbor_map AS (
    SELECT lt.dev_id,
           lt.line_id,
           lt.position_in_line,
           lt.line_length,
           u1.dev_id AS up1_dev_id,
           u2.dev_id AS up2_dev_id,
           d1.dev_id AS dn1_dev_id,
           d2.dev_id AS dn2_dev_id
    FROM line_topo lt
    LEFT JOIN line_topo u1 ON u1.line_id = lt.line_id AND u1.position_in_line = lt.position_in_line - 1
    LEFT JOIN line_topo u2 ON u2.line_id = lt.line_id AND u2.position_in_line = lt.position_in_line - 2
    LEFT JOIN line_topo d1 ON d1.line_id = lt.line_id AND d1.position_in_line = lt.position_in_line + 1
    LEFT JOIN line_topo d2 ON d2.line_id = lt.line_id AND d2.position_in_line = lt.position_in_line + 2
),
base_stops AS (
    SELECT sr.id,
           sr.dev_id,
           sr.cod_state_id,
           sr.user_id,
           sr.date_s,
           EXTRACT(HOUR FROM sr.date_s)::int AS hour_of_day,
           EXTRACT(ISODOW FROM sr.date_s)::int AS day_of_week,
           EXTRACT(MONTH FROM sr.date_s)::int AS month,
           EXTRACT(EPOCH FROM (sr.date_f - sr.date_s)) AS duration_seconds,
           LAG(sr.cod_state_id, 1) OVER (PARTITION BY sr.dev_id ORDER BY sr.date_s) AS prev_cod_state_id,
           LAG(sr.cod_state_id, 2) OVER (PARTITION BY sr.dev_id ORDER BY sr.date_s) AS prev2_cod_state_id,
           EXTRACT(EPOCH FROM (sr.date_s - LAG(sr.date_f, 1) OVER (PARTITION BY sr.dev_id ORDER BY sr.date_s))) AS seconds_since_last_stop
    FROM j_s_reg sr
    JOIN j_device d ON sr.dev_id = d.id
    JOIN j_cod_state cs ON sr.cod_state_id = cs.id
    WHERE d.client_id = :client_id
      AND d.visible = 1
      AND sr.cod_state_id IS NOT NULL
      AND sr.date_s IS NOT NULL
      AND sr.date_f IS NOT NULL
      AND sr.date_f > sr.date_s
      AND EXTRACT(EPOCH FROM (sr.date_f - sr.date_s)) > 0
      AND EXTRACT(EPOCH FROM (sr.date_f - sr.date_s)) < 86400
      AND cs.enable = 1
)
SELECT b.id,
       b.dev_id,
       b.cod_state_id,
       b.user_id,
       b.date_s,
       b.hour_of_day,
       b.day_of_week,
       b.month,
       b.duration_seconds,
       b.prev_cod_state_id,
       b.prev2_cod_state_id,
       b.seconds_since_last_stop,
       COALESCE(nm.line_id, -1) AS line_id,
       COALESCE(nm.position_in_line, -1) AS position_in_line,
       COALESCE(nm.line_length, -1) AS line_length,
       up1.cod_state_id AS up1_cod_state_id,
       CASE WHEN up1.cod_state_id IS NOT NULL THEN 1 ELSE 0 END AS up1_is_stopped,
       up1.stop_delta_sec AS up1_delta_sec,
       up2.cod_state_id AS up2_cod_state_id,
       CASE WHEN up2.cod_state_id IS NOT NULL THEN 1 ELSE 0 END AS up2_is_stopped,
       up2.stop_delta_sec AS up2_delta_sec,
       dn1.cod_state_id AS dn1_cod_state_id,
       CASE WHEN dn1.cod_state_id IS NOT NULL THEN 1 ELSE 0 END AS dn1_is_stopped,
       dn1.stop_delta_sec AS dn1_delta_sec,
       dn2.cod_state_id AS dn2_cod_state_id,
       CASE WHEN dn2.cod_state_id IS NOT NULL THEN 1 ELSE 0 END AS dn2_is_stopped,
       dn2.stop_delta_sec AS dn2_delta_sec
FROM base_stops b
LEFT JOIN neighbor_map nm ON nm.dev_id = b.dev_id
LEFT JOIN LATERAL (
    SELECT s2.cod_state_id,
           EXTRACT(EPOCH FROM (b.date_s - s2.date_s)) AS stop_delta_sec
    FROM j_s_reg s2
    WHERE s2.dev_id = nm.up1_dev_id
      AND s2.date_s <= b.date_s
      AND s2.date_s >= b.date_s - INTERVAL '300 seconds'
    ORDER BY s2.date_s DESC LIMIT 1
) up1 ON true
LEFT JOIN LATERAL (
    SELECT s2.cod_state_id,
           EXTRACT(EPOCH FROM (b.date_s - s2.date_s)) AS stop_delta_sec
    FROM j_s_reg s2
    WHERE s2.dev_id = nm.up2_dev_id
      AND s2.date_s <= b.date_s
      AND s2.date_s >= b.date_s - INTERVAL '300 seconds'
    ORDER BY s2.date_s DESC LIMIT 1
) up2 ON true
LEFT JOIN LATERAL (
    SELECT s2.cod_state_id,
           EXTRACT(EPOCH FROM (b.date_s - s2.date_s)) AS stop_delta_sec
    FROM j_s_reg s2
    WHERE s2.dev_id = nm.dn1_dev_id
      AND s2.date_s <= b.date_s
      AND s2.date_s >= b.date_s - INTERVAL '300 seconds'
    ORDER BY s2.date_s DESC LIMIT 1
) dn1 ON true
LEFT JOIN LATERAL (
    SELECT s2.cod_state_id,
           EXTRACT(EPOCH FROM (b.date_s - s2.date_s)) AS stop_delta_sec
    FROM j_s_reg s2
    WHERE s2.dev_id = nm.dn2_dev_id
      AND s2.date_s <= b.date_s
      AND s2.date_s >= b.date_s - INTERVAL '300 seconds'
    ORDER BY s2.date_s DESC LIMIT 1
) dn2 ON true
ORDER BY b.date_s
""")


@click.group()
def cli():
    """Stop Reason Suggester — predict the most likely stop reasons."""
    pass


@cli.command()
@click.option("--client-id", required=True, type=int, help="Client ID to extract data for.")
@click.option("--output", required=True, type=click.Path(), help="Output Parquet file path.")
@click.option("--database-url", default=None, help="Database URL (or set DATABASE_URL env var).")
@click.option("--min-samples", default=10, type=int, help="Min occurrences for a stop reason to be kept.")
@click.option("--verbose", is_flag=True, help="Print detailed progress.")
def extract(client_id, output, database_url, min_samples, verbose):
    """Extract historical stop data from the database."""
    engine = get_engine(database_url)

    if verbose:
        click.echo(f"Extracting data for client_id={client_id}...")

    with engine.connect() as conn:
        df = pd.read_sql(EXTRACT_SQL, conn, params={"client_id": client_id})

    if verbose:
        click.echo(f"Raw rows: {len(df)}")

    # Post-processing: compute n_neighbors_stopped
    df["n_neighbors_stopped"] = (
        df["up1_is_stopped"] + df["up2_is_stopped"] +
        df["dn1_is_stopped"] + df["dn2_is_stopped"]
    )

    # Filter rare cod_state_ids
    counts = df["cod_state_id"].value_counts()
    valid_codes = counts[counts >= min_samples].index
    before = len(df)
    df = df[df["cod_state_id"].isin(valid_codes)].copy()
    if verbose:
        click.echo(f"Filtered {before - len(df)} rows with cod_state_id having < {min_samples} samples. Remaining: {len(df)}")

    # Drop date_s (not a feature) and id
    df = df.drop(columns=["id", "date_s"])

    # Summary
    if verbose:
        click.echo(f"\n--- Summary ---")
        click.echo(f"Rows: {len(df)}")
        click.echo(f"Features: {list(df.columns)}")
        click.echo(f"Unique stop reasons: {df['cod_state_id'].nunique()}")
        click.echo(f"Unique devices: {df['dev_id'].nunique()}")
        click.echo(f"\nTop 15 stop reasons:")
        top = df["cod_state_id"].value_counts().head(15)
        for code, count in top.items():
            pct = count / len(df) * 100
            click.echo(f"  {code}: {count:>6d} ({pct:.1f}%)")
        n_with_neighbors = (df["n_neighbors_stopped"] > 0).sum()
        click.echo(f"\nCascade stats: {n_with_neighbors} stops ({n_with_neighbors/len(df)*100:.1f}%) had >= 1 neighbor stopped within 5 min")

    # Save
    outpath = Path(output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outpath, index=False)
    click.echo(f"Saved {len(df)} rows to {outpath}")


# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

DEVICE_ONLY_FEATURES = [
    "dev_id", "hour_of_day", "day_of_week", "month", "duration_seconds",
    "prev_cod_state_id", "prev2_cod_state_id", "seconds_since_last_stop",
    "user_id",
]

ALL_FEATURES = DEVICE_ONLY_FEATURES + [
    "line_id", "position_in_line", "line_length",
    "up1_cod_state_id", "up1_is_stopped", "up1_delta_sec",
    "up2_cod_state_id", "up2_is_stopped", "up2_delta_sec",
    "dn1_cod_state_id", "dn1_is_stopped", "dn1_delta_sec",
    "dn2_cod_state_id", "dn2_is_stopped", "dn2_delta_sec",
    "n_neighbors_stopped",
]

CATEGORICAL_FEATURES_A = [
    "dev_id", "prev_cod_state_id", "prev2_cod_state_id", "user_id",
]

CATEGORICAL_FEATURES_B = CATEGORICAL_FEATURES_A + [
    "line_id",
    "up1_cod_state_id", "up2_cod_state_id",
    "dn1_cod_state_id", "dn2_cod_state_id",
]


@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Input Parquet file.")
@click.option("--output", required=True, type=click.Path(), help="Output model directory.")
@click.option("--verbose", is_flag=True, help="Print detailed progress.")
def train(data, output, verbose):
    """Train a LightGBM model on extracted stop data."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report

    df = pd.read_parquet(data)
    if verbose:
        click.echo(f"Loaded {len(df)} rows, {df['cod_state_id'].nunique()} unique stop reasons")

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df["cod_state_id"])
    n_classes = len(le.classes_)
    if verbose:
        click.echo(f"Encoded {n_classes} classes")

    # Fill NaN in lag/neighbor features with -1
    fill_cols = [
        "prev_cod_state_id", "prev2_cod_state_id", "seconds_since_last_stop",
        "up1_cod_state_id", "up1_delta_sec",
        "up2_cod_state_id", "up2_delta_sec",
        "dn1_cod_state_id", "dn1_delta_sec",
        "dn2_cod_state_id", "dn2_delta_sec",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # Split
    X_a = df[DEVICE_ONLY_FEATURES].copy()
    X_b = df[ALL_FEATURES].copy()

    X_a_train, X_a_test, y_train, y_test = train_test_split(
        X_a, y, test_size=0.2, stratify=y, random_state=42
    )
    X_b_train, X_b_test, _, _ = train_test_split(
        X_b, y, test_size=0.2, stratify=y, random_state=42
    )

    # Cast categorical columns to 'category' dtype
    for col in CATEGORICAL_FEATURES_A:
        X_a_train[col] = X_a_train[col].astype("category")
        X_a_test[col] = X_a_test[col].astype("category")

    for col in CATEGORICAL_FEATURES_B:
        X_b_train[col] = X_b_train[col].astype("category")
        X_b_test[col] = X_b_test[col].astype("category")

    params = dict(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=42,
        n_jobs=-1,
    )

    # Train Model A (device-only)
    if verbose:
        click.echo("\n--- Training Model A (device-only) ---")
    model_a = LGBMClassifier(**params)
    model_a.fit(X_a_train, y_train, categorical_feature=CATEGORICAL_FEATURES_A)

    # Train Model B (device+line)
    if verbose:
        click.echo("--- Training Model B (device+line) ---")
    model_b = LGBMClassifier(**params)
    model_b.fit(X_b_train, y_train, categorical_feature=CATEGORICAL_FEATURES_B)

    # Evaluate both
    def evaluate_model(model, X_test, y_test, name, features):
        proba = model.predict_proba(X_test)
        top1 = (proba.argmax(axis=1) == y_test).mean()
        top3 = top_k_accuracy(proba, y_test, k=3)
        top5 = top_k_accuracy(proba, y_test, k=5)

        click.echo(f"\n{'='*50}")
        click.echo(f"  {name}")
        click.echo(f"{'='*50}")
        click.echo(f"  Top-1 accuracy: {top1:.4f}")
        click.echo(f"  Top-3 accuracy: {top3:.4f}")
        click.echo(f"  Top-5 accuracy: {top5:.4f}")

        # Feature importance
        importances = model.feature_importances_
        feat_imp = sorted(zip(features, importances), key=lambda x: -x[1])
        click.echo(f"\n  Feature importance:")
        for fname, imp in feat_imp[:10]:
            click.echo(f"    {fname:30s} {imp:>6d}")

        # Classification report for top 10 reasons
        if verbose:
            y_pred = proba.argmax(axis=1)
            top10_classes = np.argsort(np.bincount(y_test))[-10:]
            mask = np.isin(y_test, top10_classes)
            if mask.sum() > 0:
                target_names = [str(le.classes_[c]) for c in top10_classes]
                click.echo(f"\n  Classification report (top 10 reasons):")
                report = classification_report(
                    y_test[mask], y_pred[mask],
                    labels=top10_classes, target_names=target_names,
                    zero_division=0,
                )
                for line in report.split("\n"):
                    click.echo(f"    {line}")

        return top1, top3, top5

    click.echo("\n" + "=" * 60)
    click.echo("  MODEL COMPARISON")
    click.echo("=" * 60)
    t1_a, t3_a, t5_a = evaluate_model(model_a, X_a_test, y_test, "Model A (device-only)", DEVICE_ONLY_FEATURES)
    t1_b, t3_b, t5_b = evaluate_model(model_b, X_b_test, y_test, "Model B (device+line)", ALL_FEATURES)

    click.echo(f"\n{'='*50}")
    click.echo(f"  Side-by-side")
    click.echo(f"{'='*50}")
    click.echo(f"  {'Metric':<20s} {'Model A':>10s} {'Model B':>10s} {'Diff':>10s}")
    click.echo(f"  {'Top-1':<20s} {t1_a:>10.4f} {t1_b:>10.4f} {t1_b-t1_a:>+10.4f}")
    click.echo(f"  {'Top-3':<20s} {t3_a:>10.4f} {t3_b:>10.4f} {t3_b-t3_a:>+10.4f}")
    click.echo(f"  {'Top-5':<20s} {t5_a:>10.4f} {t5_b:>10.4f} {t5_b-t5_a:>+10.4f}")

    # Save Model B (full features)
    outdir = Path(output)
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_b, outdir / "model.joblib")
    joblib.dump(le, outdir / "label_encoder.joblib")

    metadata = {
        "features": ALL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES_B,
        "n_classes": n_classes,
        "classes": [int(c) for c in le.classes_],
        "metrics": {
            "model_a": {"top1": t1_a, "top3": t3_a, "top5": t5_a},
            "model_b": {"top1": t1_b, "top3": t3_b, "top5": t5_b},
        },
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"\nModel saved to {outdir}/")


def top_k_accuracy(proba, y_true, k):
    """Compute top-k accuracy from probability matrix."""
    from sklearn.metrics import top_k_accuracy_score
    return top_k_accuracy_score(y_true, proba, k=k, labels=np.arange(proba.shape[1]))


# ---------------------------------------------------------------------------
# PREDICT — reusable functions + CLI command
# ---------------------------------------------------------------------------

DEVICE_INFO_SQL = text("""
SELECT d.id AS dev_id, d.name AS dev_name,
       d.line_id, d.index AS position_in_line,
       l.name AS line_name,
       (SELECT COUNT(*) FROM j_device d2
        WHERE d2.line_id = d.line_id AND d2.visible = 1 AND d2.index IS NOT NULL) AS line_length
FROM j_device d
LEFT JOIN j_line l ON d.line_id = l.id
WHERE d.id = :dev_id
""")

RECENT_STOPS_SQL = text("""
SELECT sr.cod_state_id, sr.date_s, sr.date_f
FROM j_s_reg sr
JOIN j_cod_state cs ON sr.cod_state_id = cs.id
WHERE sr.dev_id = :dev_id
  AND sr.cod_state_id IS NOT NULL
  AND sr.date_s IS NOT NULL
  AND sr.date_f IS NOT NULL
  AND sr.date_s < :before
  AND cs.enable = 1
ORDER BY sr.date_s DESC
LIMIT 3
""")

NEIGHBOR_DEVS_SQL = text("""
SELECT d.id AS dev_id, d.index AS position_in_line
FROM j_device d
WHERE d.line_id = :line_id
  AND d.visible = 1
  AND d.index IS NOT NULL
  AND d.index IN (:idx_up1, :idx_up2, :idx_dn1, :idx_dn2)
""")

NEIGHBOR_STOP_SQL = text("""
SELECT s.cod_state_id, s.date_s
FROM j_s_reg s
WHERE s.dev_id = :dev_id
  AND s.date_s IS NOT NULL
  AND s.date_s < :before
  AND s.date_s >= :before - INTERVAL '300 seconds'
ORDER BY s.date_s DESC
LIMIT 1
""")

REASON_NAMES_SQL = text("""
SELECT cs.id, cs.description, cs.code_f, cs.code_t
FROM j_cod_state cs
WHERE cs.id = ANY(:ids)
""")


def load_model_artifacts(model_dir):
    """Load model, label encoder, and metadata from a model directory.

    Returns dict with keys: model, label_encoder, metadata.
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    label_encoder = joblib.load(model_dir / "label_encoder.joblib")
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    return {
        "model": model,
        "label_encoder": label_encoder,
        "metadata": metadata,
    }


def predict_stop_reasons(engine, artifacts, dev_id, duration=60.0, user_id=None, top_k=3, timestamp=None):
    """Predict top-k stop reasons for a device.

    Args:
        engine: SQLAlchemy engine.
        artifacts: Dict from load_model_artifacts().
        dev_id: Device ID.
        duration: Current stop duration in seconds.
        user_id: Operator user ID (optional).
        top_k: Number of predictions to return.
        timestamp: When the stop started (ISO string or datetime). Defaults to now.

    Returns:
        Dict with keys: device, predictions, context.

    Raises:
        ValueError: If device not found.
    """
    model = artifacts["model"]
    le = artifacts["label_encoder"]
    metadata = artifacts["metadata"]
    features = metadata["features"]
    if timestamp and isinstance(timestamp, str):
        now = datetime.fromisoformat(timestamp)
    elif timestamp:
        now = timestamp
    else:
        now = datetime.now()

    with engine.connect() as conn:
        # Device info
        dev_row = conn.execute(DEVICE_INFO_SQL, {"dev_id": dev_id}).fetchone()
        if not dev_row:
            raise ValueError(f"Device {dev_id} not found")

        dev_info = dict(dev_row._mapping)

        # Recent stops for LAG features
        recent = conn.execute(RECENT_STOPS_SQL, {"dev_id": dev_id, "before": now}).fetchall()

        prev_cod_state_id = -1
        prev2_cod_state_id = -1
        seconds_since_last_stop = -1

        if len(recent) >= 1:
            prev_cod_state_id = recent[0]._mapping["cod_state_id"]
            if recent[0]._mapping["date_f"] is not None:
                seconds_since_last_stop = (now - recent[0]._mapping["date_f"]).total_seconds()
        if len(recent) >= 2:
            prev2_cod_state_id = recent[1]._mapping["cod_state_id"]

        # Neighbor features
        position = dev_info.get("position_in_line")
        line_id = dev_info.get("line_id")

        neighbor_features = {
            "up1_cod_state_id": -1, "up1_is_stopped": 0, "up1_delta_sec": -1,
            "up2_cod_state_id": -1, "up2_is_stopped": 0, "up2_delta_sec": -1,
            "dn1_cod_state_id": -1, "dn1_is_stopped": 0, "dn1_delta_sec": -1,
            "dn2_cod_state_id": -1, "dn2_is_stopped": 0, "dn2_delta_sec": -1,
        }

        if line_id is not None and position is not None:
            neighbor_offsets = {
                "up1": position - 1,
                "up2": position - 2,
                "dn1": position + 1,
                "dn2": position + 2,
            }
            neighbor_rows = conn.execute(NEIGHBOR_DEVS_SQL, {
                "line_id": line_id,
                "idx_up1": neighbor_offsets["up1"],
                "idx_up2": neighbor_offsets["up2"],
                "idx_dn1": neighbor_offsets["dn1"],
                "idx_dn2": neighbor_offsets["dn2"],
            }).fetchall()

            idx_to_dev = {r._mapping["position_in_line"]: r._mapping["dev_id"] for r in neighbor_rows}

            for prefix, offset in neighbor_offsets.items():
                neighbor_dev_id = idx_to_dev.get(offset)
                if neighbor_dev_id:
                    ns = conn.execute(NEIGHBOR_STOP_SQL, {"dev_id": neighbor_dev_id, "before": now}).fetchone()
                    if ns:
                        delta = (now - ns._mapping["date_s"]).total_seconds()
                        neighbor_features[f"{prefix}_cod_state_id"] = ns._mapping["cod_state_id"]
                        neighbor_features[f"{prefix}_is_stopped"] = 1
                        neighbor_features[f"{prefix}_delta_sec"] = delta

        n_neighbors_stopped = sum(
            neighbor_features[f"{p}_is_stopped"] for p in ["up1", "up2", "dn1", "dn2"]
        )

        # Build feature vector
        feature_values = {
            "dev_id": dev_id,
            "hour_of_day": now.hour,
            "day_of_week": now.isoweekday(),
            "month": now.month,
            "duration_seconds": duration,
            "prev_cod_state_id": prev_cod_state_id,
            "prev2_cod_state_id": prev2_cod_state_id,
            "seconds_since_last_stop": seconds_since_last_stop,
            "user_id": user_id if user_id is not None else -1,
            "line_id": line_id if line_id is not None else -1,
            "position_in_line": position if position is not None else -1,
            "line_length": dev_info.get("line_length") or -1,
            **neighbor_features,
            "n_neighbors_stopped": n_neighbors_stopped,
        }

        X = pd.DataFrame([feature_values])[features]

        # Cast categoricals
        for col in metadata["categorical_features"]:
            X[col] = X[col].astype("category")

        # Predict
        proba = model.predict_proba(X)[0]
        top_indices = np.argsort(proba)[::-1][:top_k]

        cod_state_ids = [int(le.classes_[i]) for i in top_indices]
        confidences = [float(proba[i]) for i in top_indices]

        # Get reason names
        reason_names = {}
        if cod_state_ids:
            rows = conn.execute(REASON_NAMES_SQL, {"ids": cod_state_ids}).fetchall()
            reason_names = {
                r._mapping["id"]: {
                    "description": r._mapping["description"],
                    "code_f": r._mapping["code_f"],
                    "code_t": r._mapping["code_t"],
                }
                for r in rows
            }

    return {
        "device": {
            "id": dev_id,
            "name": dev_info["dev_name"],
            "line": dev_info.get("line_name"),
            "line_id": dev_info.get("line_id"),
            "position": dev_info.get("position_in_line"),
            "line_length": dev_info.get("line_length"),
        },
        "predictions": [
            {
                "rank": rank,
                "cod_state_id": code,
                "description": reason_names.get(code, {}).get("description", "Unknown"),
                "code_f": reason_names.get(code, {}).get("code_f"),
                "code_t": reason_names.get(code, {}).get("code_t"),
                "confidence": conf,
            }
            for rank, (code, conf) in enumerate(zip(cod_state_ids, confidences), 1)
        ],
        "context": {
            "timestamp": now.isoformat(timespec="seconds"),
            "hour_of_day": now.hour,
            "day_of_week": now.isoweekday(),
            "prev_cod_state_id": prev_cod_state_id,
            "prev2_cod_state_id": prev2_cod_state_id,
            "seconds_since_last_stop": seconds_since_last_stop,
            "neighbors_stopped": n_neighbors_stopped,
        },
    }


@cli.command()
@click.option("--model-dir", required=True, type=click.Path(exists=True), help="Model directory.")
@click.option("--dev-id", required=True, type=int, help="Device ID to predict for.")
@click.option("--duration", default=60.0, type=float, help="Current stop duration in seconds (default 60).")
@click.option("--user-id", default=None, type=int, help="Operator user ID (optional).")
@click.option("--top-k", default=5, type=int, help="Number of predictions to show.")
@click.option("--database-url", default=None, help="Database URL (or set DATABASE_URL env var).")
@click.option("--verbose", is_flag=True, help="Print detailed context.")
def predict(model_dir, dev_id, duration, user_id, top_k, database_url, verbose):
    """Predict top-k stop reasons for a device."""
    artifacts = load_model_artifacts(model_dir)
    engine = get_engine(database_url)

    try:
        result = predict_stop_reasons(engine, artifacts, dev_id, duration, user_id, top_k)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    dev = result["device"]
    preds = result["predictions"]
    ctx = result["context"]

    if verbose:
        click.echo(f"Device: {dev['name']} (id={dev['id']})")
        click.echo(f"Line: {dev.get('line', 'N/A')} (id={dev.get('line_id', 'N/A')})")
        click.echo(f"Position: {dev.get('position', 'N/A')}, Line length: {dev.get('line_length', 'N/A')}")

    click.echo(f"\n{'='*70}")
    click.echo(f"  Top-{top_k} predicted stop reasons for device {dev['name']} (id={dev['id']})")
    click.echo(f"{'='*70}")
    click.echo(f"  {'Rank':<6s} {'Conf':>7s}  {'Code':>6s}  {'Reason'}")
    click.echo(f"  {'-'*6} {'-'*7}  {'-'*6}  {'-'*40}")

    for p in preds:
        click.echo(f"  {p['rank']:<6d} {p['confidence']:>6.1%}  {p['cod_state_id']:>6d}  {p['description']}")

    if verbose:
        click.echo(f"\n  Context:")
        click.echo(f"    Time: {ctx['timestamp']} (hour={ctx['hour_of_day']}, dow={ctx['day_of_week']})")
        click.echo(f"    Duration: {duration:.0f}s")
        click.echo(f"    Previous reason: cod_state_id={ctx['prev_cod_state_id']}")
        click.echo(f"    Neighbors stopped: {ctx['neighbors_stopped']}/4")


# ---------------------------------------------------------------------------
# SERVE
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--model-dir", required=True, type=click.Path(exists=True), help="Model directory.")
@click.option("--database-url", default=None, help="Database URL (or set DATABASE_URL env var).")
@click.option("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
@click.option("--port", default=8008, type=int, help="Bind port (default 8008).")
def serve(model_dir, database_url, host, port):
    """Start the FastAPI prediction server."""
    import uvicorn

    # Resolve and validate before passing to uvicorn
    model_dir = str(Path(model_dir).resolve())
    database_url = database_url or os.environ.get("DATABASE_URL")
    if not database_url:
        click.echo("Error: DATABASE_URL not set. Pass --database-url or set the env var.", err=True)
        sys.exit(1)

    os.environ["MODEL_DIR"] = model_dir
    os.environ["DATABASE_URL"] = database_url

    click.echo(f"Starting prediction server on {host}:{port}")
    click.echo(f"Model: {model_dir}")
    uvicorn.run("api:app", host=host, port=port)


if __name__ == "__main__":
    cli()
