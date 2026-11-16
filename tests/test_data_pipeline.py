from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import category_summary, city_summary, summary_metrics
from src.data_cleaning import clean_restaurants, parse_categories
from src.ml_model import save_model_bundle, train_and_evaluate_models


def _restaurant_frame(count: int = 80) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "business_id": f"business-{index}",
                "name": f"Restaurant {index}",
                "city": "Phoenix" if index % 3 else "Cleveland",
                "state": "AZ" if index % 3 else "OH",
                "rating": [2.5, 3.0, 3.5, 4.0, 4.5, 5.0][index % 6],
                "review_count": (index * 17) % 230,
                "categories": (
                    "['Restaurants', 'Pizza', 'Italian']"
                    if index % 2
                    else '["Restaurants", "Mexican"]'
                ),
            }
        )
    return pd.DataFrame(rows)


def test_category_parser_handles_legacy_and_json_lists() -> None:
    assert parse_categories("[u'Restaurants', u'Pizza', u'Pizza']") == [
        "Restaurants",
        "Pizza",
    ]
    assert parse_categories('["Mexican", "Fast Food"]') == ["Mexican", "Fast Food"]
    assert parse_categories("Cafes; Breakfast & Brunch") == ["Cafes", "Breakfast & Brunch"]


def test_cleaner_drops_bad_ratings_and_duplicate_ids_and_keeps_city_region() -> None:
    frame = pd.DataFrame(
        {
            "business_id": ["same", "same", "missing-rating", "bad-rating", "other-city"],
            "name": ["First", "Duplicate", "No rating", "Bad rating", "Other"],
            "city": ["Montréal", "Montréal", "Phoenix", "Phoenix", "Phoenix"],
            "state": ["qc", "qc", "AZ", "AZ", "OH"],
            "stars": [4.0, 5.0, np.nan, 6.0, 3.5],
            "review_count": [20, 99, 4, 8, None],
            "categories": ["['Pizza']", "['Pizza']", "['Pizza']", "['Pizza']", "['Mexican']"],
        }
    )
    cleaned, report = clean_restaurants(frame)

    assert len(cleaned) == 2
    assert report["duplicate_ids_removed"] == 1
    assert report["missing_ratings_removed"] == 1
    assert report["invalid_ratings_removed"] == 1
    assert set(cleaned["location"]) == {"Montreal, QC", "Phoenix, OH"}
    assert cleaned["review_count"].min() == 0


def test_analysis_counts_and_competition_ranks_are_market_specific() -> None:
    restaurants, _ = clean_restaurants(_restaurant_frame())
    metrics = summary_metrics(restaurants)
    locations = city_summary(restaurants)
    categories = category_summary(
        restaurants,
        location="Phoenix, AZ",
        minimum_restaurants=2,
    )

    assert metrics["total_restaurants"] == 80
    assert metrics["number_of_cities"] == 2
    assert len(locations) == 2
    assert set(categories["category"]) == {"Pizza", "Italian", "Mexican"}
    assert categories["competition_score"].between(0, 100).all()
    assert categories["opportunity_index"].between(0, 100).all()


def test_rating_model_is_evaluated_on_a_holdout_and_bundle_can_be_saved(tmp_path) -> None:
    restaurants, _ = clean_restaurants(_restaurant_frame())
    result = train_and_evaluate_models(restaurants)
    metrics = result["metrics"]

    assert metrics["train_rows"] == 64
    assert metrics["test_rows"] == 16
    assert np.isfinite(metrics["mae"])
    assert np.isfinite(metrics["rmse"])
    assert -1 <= metrics["silhouette_score"] <= 1
    assert len(result["predictions"]) == 16

    model_path = save_model_bundle(result, tmp_path)
    assert model_path.exists()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["test_rows"] == 16


def test_upload_requires_a_rating_column() -> None:
    with pytest.raises(ValueError, match="rating column is required"):
        clean_restaurants(pd.DataFrame({"name": ["No rating"]}))
