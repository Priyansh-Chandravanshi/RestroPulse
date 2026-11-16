"""Load and clean restaurant CSV and Yelp JSON Lines data."""

from __future__ import annotations

import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_COLUMN_ALIASES = {
    "id": "business_id",
    "businessid": "business_id",
    "restaurantid": "business_id",
    "restaurant_name": "name",
    "restaurantname": "name",
    "category": "categories",
    "restaurant_categories": "categories",
    "reviewcount": "review_count",
    "num_reviews": "review_count",
    "number_of_reviews": "review_count",
    "rating": "stars",
    "average_rating": "stars",
    "avg_rating": "stars",
    "star_rating": "stars",
    "province": "state",
    "state_code": "state",
}

_GENERIC_CATEGORIES = {"restaurants", "food"}


def _column_name(value: Any) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    return _COLUMN_ALIASES.get(name, _COLUMN_ALIASES.get(name.replace("_", ""), name))


def _clean_text(value: Any, default: str) -> str:
    if pd.isna(value):
        return default
    text = " ".join(str(value).strip().split())
    if not text:
        return default
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def parse_categories(value: Any) -> list[str]:
    """Parse category lists from Python literals, JSON arrays, or delimited CSV cells."""
    if value is None or (not isinstance(value, (list, tuple, set)) and pd.isna(value)):
        return []

    parsed: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text[:1] in "[(":
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(text)
                except (ValueError, json.JSONDecodeError):
                    parsed = re.split(r"\s*[;,|]\s*", text.strip("[]() "))
        else:
            parsed = re.split(r"\s*[;,|]\s*", text)

    if not isinstance(parsed, (list, tuple, set)):
        parsed = [parsed]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if item is None or pd.isna(item):
            continue
        category = " ".join(str(item).strip().split())
        if not category:
            continue
        key = category.casefold()
        if key not in seen:
            cleaned.append(category)
            seen.add(key)
    return cleaned


def _prepare_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_map: dict[Any, str] = {}
    for original in result.columns:
        normalized = _column_name(original)
        if (
            normalized != original
            and normalized not in result.columns
            and normalized not in rename_map.values()
        ):
            rename_map[original] = normalized
    result = result.rename(columns=rename_map)

    # Prefer explicit city/state columns, but split a combined location if needed.
    if "city" not in result.columns and "location" in result.columns:
        split = result["location"].fillna("").astype(str).str.rsplit(",", n=1, expand=True)
        result["city"] = split[0]
        if split.shape[1] > 1 and "state" not in result.columns:
            result["state"] = split[1]

    return result


def clean_restaurants(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Normalize fields, remove duplicate IDs and discard rows without valid ratings."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Restaurant data must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("The uploaded dataset is empty.")

    data = _prepare_columns(frame)
    if "stars" not in data.columns:
        raise ValueError("A rating column is required. Use stars, rating, or average_rating.")

    input_rows = len(data)
    if "business_id" not in data.columns:
        data["business_id"] = [f"row-{index}" for index in range(len(data))]
    else:
        missing_ids = data["business_id"].isna() | data["business_id"].astype(str).str.strip().eq("")
        data.loc[missing_ids, "business_id"] = [f"row-{index}" for index in data.index[missing_ids]]
        data["business_id"] = data["business_id"].astype(str).str.strip()

    for column, default in (
        ("name", "Unnamed restaurant"),
        ("city", "Unknown city"),
        ("state", "Unknown region"),
    ):
        if column not in data.columns:
            data[column] = default
        data[column] = data[column].map(lambda value: _clean_text(value, default))

    if "categories" not in data.columns:
        data["categories"] = [[] for _ in range(len(data))]
    data["categories_clean"] = data["categories"].map(parse_categories)
    data["categories_display"] = data["categories_clean"].map(lambda values: ", ".join(values))
    data["category_count"] = data["categories_clean"].map(len).astype(int)

    if "review_count" not in data.columns:
        data["review_count"] = 0
    data["review_count"] = pd.to_numeric(data["review_count"], errors="coerce").fillna(0)
    data["review_count"] = data["review_count"].clip(lower=0).astype(int)
    data["stars"] = pd.to_numeric(data["stars"], errors="coerce")

    missing_rating = int(data["stars"].isna().sum())
    invalid_rating = int((data["stars"].notna() & ~data["stars"].between(1, 5)).sum())
    data = data[data["stars"].between(1, 5)].copy()

    before_deduplication = len(data)
    data = data.drop_duplicates(subset="business_id", keep="first").copy()
    duplicate_rows = before_deduplication - len(data)
    if data.empty:
        raise ValueError("No rows have a valid restaurant rating between 1 and 5.")

    # Canonicalize common location spelling/case differences without mapping a state to a city.
    data["city"] = data["city"].map(lambda value: value.title() if value != "Unknown city" else value)
    data["state"] = data["state"].map(
        lambda value: value.upper() if len(value) <= 3 else value.title()
    )
    data["location"] = data.apply(
        lambda row: row["city"]
        if row["state"] == "Unknown region"
        else f"{row['city']}, {row['state']}",
        axis=1,
    )
    data["log_review_count"] = np.log1p(data["review_count"])
    data = data.sort_values(["location", "name"], kind="stable").reset_index(drop=True)

    report = {
        "input_rows": input_rows,
        "retained_rows": len(data),
        "missing_ratings_removed": missing_rating,
        "invalid_ratings_removed": invalid_rating,
        "duplicate_ids_removed": duplicate_rows,
        "rows_removed": input_rows - len(data),
    }
    return data, report


def load_csv(path_or_buffer: Any) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read and clean a CSV path or file-like upload."""
    try:
        return clean_restaurants(pd.read_csv(path_or_buffer))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read this CSV file: {exc}") from exc


def load_yelp_json(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the cleaned restaurant table from Yelp business JSON Lines."""
    source = pd.read_json(path, lines=True)
    if "is_open" in source.columns:
        source = source[source["is_open"].eq(1)].copy()
    if "categories" in source.columns:
        source["categories"] = source["categories"].map(parse_categories)
        source = source[
            source["categories"].map(
                lambda categories: any(category.casefold() == "restaurants" for category in categories)
            )
        ].copy()
    keep = [
        column
        for column in (
            "business_id",
            "name",
            "city",
            "state",
            "stars",
            "review_count",
            "categories",
        )
        if column in source.columns
    ]
    return clean_restaurants(source[keep])


def analysis_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Return restaurant rows with generic directory categories removed."""
    result = frame.copy()
    result["analysis_categories"] = result["categories_clean"].map(
        lambda values: [
            category
            for category in values
            if category.casefold() not in _GENERIC_CATEGORIES
        ]
    )
    return result
