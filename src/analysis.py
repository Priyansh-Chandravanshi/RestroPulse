"""Reusable descriptive and competition analysis for RestroPulse."""

from __future__ import annotations

import pandas as pd

from .data_cleaning import analysis_categories


def summary_metrics(restaurants: pd.DataFrame) -> dict[str, float | int]:
    """Return headline metrics for the selected restaurant records."""
    return {
        "total_restaurants": int(restaurants["business_id"].nunique()),
        "average_rating": float(restaurants["stars"].mean()) if len(restaurants) else 0.0,
        "total_reviews": int(restaurants["review_count"].sum()),
        "number_of_cities": int(restaurants["location"].nunique()),
    }


def city_summary(restaurants: pd.DataFrame) -> pd.DataFrame:
    """Aggregate restaurant count, ratings, and reviews by city plus region."""
    if restaurants.empty:
        return pd.DataFrame(
            columns=["location", "city", "state", "restaurant_count", "average_rating", "total_reviews"]
        )
    result = (
        restaurants.groupby(["location", "city", "state"], dropna=False)
        .agg(
            restaurant_count=("business_id", "nunique"),
            average_rating=("stars", "mean"),
            total_reviews=("review_count", "sum"),
            median_reviews=("review_count", "median"),
        )
        .reset_index()
        .sort_values(["restaurant_count", "location"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result


def category_summary(
    restaurants: pd.DataFrame,
    location: str | None = None,
    minimum_restaurants: int = 1,
) -> pd.DataFrame:
    """Score category supply, review activity, and a relative opportunity proxy.

    Competition scores rank categories within the selected city or dataset. The
    opportunity index is a descriptive proxy: it combines lower observed supply
    with review activity per existing venue, and is not a demand forecast.
    """
    columns = [
        "category",
        "restaurant_count",
        "restaurant_share_pct",
        "total_reviews",
        "review_share_pct",
        "average_rating",
        "review_weighted_rating",
        "reviews_per_restaurant",
        "competition_score",
        "opportunity_index",
    ]
    if restaurants.empty:
        return pd.DataFrame(columns=columns)

    selected = restaurants if location is None else restaurants[restaurants["location"].eq(location)]
    if selected.empty:
        return pd.DataFrame(columns=columns)

    category_rows = analysis_categories(selected).explode("analysis_categories")
    category_rows = category_rows[
        category_rows["analysis_categories"].notna()
        & category_rows["analysis_categories"].astype(str).str.strip().ne("")
    ].copy()
    if category_rows.empty:
        return pd.DataFrame(columns=columns)

    category_rows["weighted_rating_points"] = category_rows["stars"] * category_rows["review_count"]
    result = (
        category_rows.groupby("analysis_categories", dropna=False)
        .agg(
            restaurant_count=("business_id", "nunique"),
            total_reviews=("review_count", "sum"),
            average_rating=("stars", "mean"),
            weighted_rating_points=("weighted_rating_points", "sum"),
        )
        .rename_axis("category")
        .reset_index()
    )
    result["review_weighted_rating"] = result["weighted_rating_points"].div(
        result["total_reviews"].replace(0, pd.NA)
    ).fillna(result["average_rating"])
    result["reviews_per_restaurant"] = result["total_reviews"].div(
        result["restaurant_count"].replace(0, pd.NA)
    ).fillna(0)

    total_restaurants = max(int(selected["business_id"].nunique()), 1)
    total_reviews = max(int(selected["review_count"].sum()), 1)
    result["restaurant_share_pct"] = result["restaurant_count"].div(total_restaurants).mul(100)
    result["review_share_pct"] = result["total_reviews"].div(total_reviews).mul(100)

    # Percentile ranks make the index interpretable within the selected market.
    density_rank = result["restaurant_count"].rank(method="average", pct=True)
    engagement_rank = result["reviews_per_restaurant"].rank(method="average", pct=True)
    rating_rank = result["review_weighted_rating"].rank(method="average", pct=True)
    result["competition_score"] = (
        100 * (0.45 * density_rank + 0.35 * engagement_rank + 0.20 * rating_rank)
    )
    result["opportunity_index"] = (
        100 * (0.55 * (1 - density_rank) + 0.45 * engagement_rank)
    )

    result = result[result["restaurant_count"].ge(max(1, minimum_restaurants))]
    return (
        result[columns]
        .sort_values(["competition_score", "restaurant_count"], ascending=[False, False])
        .reset_index(drop=True)
    )


def top_restaurants(
    restaurants: pd.DataFrame,
    minimum_reviews: int = 1,
    ascending: bool = False,
    limit: int = 20,
) -> pd.DataFrame:
    """Return the highest or lowest rated restaurants above a review threshold."""
    eligible = restaurants[restaurants["review_count"].ge(max(0, minimum_reviews))].copy()
    return eligible.sort_values(
        ["stars", "review_count", "name"],
        ascending=[ascending, False, True],
    ).head(limit)


def build_market_report(
    restaurants: pd.DataFrame,
    selected_location: str | None = None,
    minimum_category_restaurants: int = 2,
) -> str:
    """Create a short, downloadable summary of a selected market."""
    scope = restaurants
    title = "All selected cities"
    if selected_location:
        scope = restaurants[restaurants["location"].eq(selected_location)]
        title = selected_location

    metrics = summary_metrics(scope)
    categories = category_summary(scope, minimum_restaurants=minimum_category_restaurants)
    low_competition = categories.sort_values(
        ["competition_score", "restaurant_count"], ascending=[True, True]
    ).head(5)
    opportunities = categories.sort_values(
        ["opportunity_index", "reviews_per_restaurant"], ascending=[False, False]
    ).head(5)

    def format_categories(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "- No categories meet the selected minimum venue count."
        return "\n".join(
            f"- {row.category}: {int(row.restaurant_count)} restaurants, "
            f"{int(row.total_reviews):,} reviews, "
            f"competition {row.competition_score:.1f}/100, "
            f"opportunity proxy {row.opportunity_index:.1f}/100"
            for row in frame.itertuples(index=False)
        )

    return f"""# RestroPulse Market Report

**Market:** {title}

## Snapshot

- Restaurants: {metrics['total_restaurants']:,}
- Average rating: {metrics['average_rating']:.2f}/5
- Customer reviews: {metrics['total_reviews']:,}
- Cities represented: {metrics['number_of_cities']:,}

## Lowest relative competition

{format_categories(low_competition)}

## Highest opportunity proxy

{format_categories(opportunities)}

## Interpretation

Competition is ranked relative to the categories observed in this dataset. The opportunity proxy combines lower observed restaurant supply with reviews per existing restaurant. Review activity is only a demand signal for businesses already present; this report cannot estimate unobserved demand, profitability, or business success. Categories overlap because one restaurant can list several categories.
"""
