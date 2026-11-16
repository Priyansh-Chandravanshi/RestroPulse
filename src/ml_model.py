"""Train and evaluate restaurant rating and clustering models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _model_features(restaurants: pd.DataFrame) -> pd.DataFrame:
    features = restaurants[
        ["business_id", "city", "state", "stars", "review_count", "category_count"]
    ].copy()
    features["log_review_count"] = np.log1p(features["review_count"])
    features["city"] = features["city"].fillna("Unknown city").astype(str)
    features["state"] = features["state"].fillna("Unknown region").astype(str)
    features["category_count"] = pd.to_numeric(
        features["category_count"], errors="coerce"
    ).fillna(0)
    return features


def train_and_evaluate_models(
    restaurants: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate linear regression on a holdout set and fit a K-Means segmenter."""
    if len(restaurants) < 20:
        raise ValueError("At least 20 valid restaurants are required for model evaluation.")

    features = _model_features(restaurants).reset_index(drop=True)
    x = features[["log_review_count", "category_count", "city", "state"]]
    y = features["stars"].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), ["log_review_count", "category_count"]),
            ("location", OneHotEncoder(handle_unknown="ignore"), ["city", "state"]),
        ],
        remainder="drop",
    )
    holdout_model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("regression", LinearRegression()),
        ]
    )
    holdout_model.fit(x_train, y_train)
    predicted = holdout_model.predict(x_test)
    baseline_prediction = np.full(len(y_test), float(y_train.median()))
    metrics: dict[str, float | int | None] = {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": float(r2_score(y_test, predicted)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline_prediction)),
    }
    baseline_mae = float(metrics["baseline_mae"])
    metrics["mae_improvement_vs_baseline_pct"] = (
        float((baseline_mae - float(metrics["mae"])) / baseline_mae * 100)
        if baseline_mae
        else 0.0
    )

    predictions = pd.DataFrame(
        {
            "business_id": features.loc[x_test.index, "business_id"].to_numpy(),
            "actual_stars": y_test.to_numpy(),
            "predicted_stars": predicted,
        }
    ).reset_index(drop=True)

    # Fit the production pipeline on all available rows only after holdout scoring.
    final_model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("regression", LinearRegression()),
        ]
    ).fit(x, y)

    cluster_features = features[["stars", "log_review_count", "category_count"]].astype(float)
    actual_clusters = min(max(2, int(n_clusters)), len(cluster_features) - 1)
    cluster_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=actual_clusters, n_init=20, random_state=random_state),
            ),
        ]
    )
    cluster_labels = cluster_pipeline.fit_predict(cluster_features) + 1
    unique_labels = np.unique(cluster_labels)
    if 1 < len(unique_labels) < len(cluster_labels):
        silhouette_sample_size = min(5000, len(cluster_labels))
        metrics["silhouette_score"] = float(
            silhouette_score(
                cluster_pipeline.named_steps["scale"].transform(cluster_features),
                cluster_labels,
                sample_size=silhouette_sample_size,
                random_state=random_state,
            )
        )
    else:
        metrics["silhouette_score"] = None
    metrics["cluster_count"] = int(len(unique_labels))

    segmented = features[
        ["business_id", "city", "state", "stars", "review_count", "category_count"]
    ].copy()
    segmented["cluster"] = cluster_labels.astype(int)
    return {
        "metrics": metrics,
        "rating_model": final_model,
        "holdout_model": holdout_model,
        "cluster_model": cluster_pipeline,
        "predictions": predictions,
        "segmented_restaurants": segmented,
    }


def model_coefficients(model: Pipeline) -> pd.DataFrame:
    """Return the fitted linear model's standardized and one-hot coefficients."""
    preprocess = model.named_steps["preprocess"]
    regression = model.named_steps["regression"]
    names = preprocess.get_feature_names_out()
    result = pd.DataFrame(
        {"feature": names, "coefficient": regression.coef_.ravel()}
    )
    result["absolute_effect"] = result["coefficient"].abs()
    return result.sort_values("absolute_effect", ascending=False).reset_index(drop=True)


def save_model_bundle(result: dict[str, Any], output_dir: str | Path) -> Path:
    """Persist evaluated pipelines and their holdout metrics."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.pkl"
    bundle = {
        "rating_model": result["rating_model"],
        "cluster_model": result["cluster_model"],
        "metrics": result["metrics"],
        "model_version": 1,
    }
    joblib.dump(bundle, model_path)
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(json.dumps(result["metrics"], indent=2, allow_nan=False), encoding="utf-8")
    return model_path


def main() -> None:
    from .data_cleaning import load_csv

    project_root = Path(__file__).resolve().parents[1]
    restaurants, report = load_csv(project_root / "data" / "cleaned.csv")
    result = train_and_evaluate_models(restaurants)
    model_path = save_model_bundle(result, project_root / "models")
    print(json.dumps({"cleaning": report, "metrics": result["metrics"], "model_path": str(model_path)}, indent=2))


if __name__ == "__main__":
    main()
