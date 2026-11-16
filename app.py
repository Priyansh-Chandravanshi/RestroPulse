"""Interactive RestroPulse restaurant trends and competition dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import (  # noqa: E402
    build_market_report,
    category_summary,
    city_summary,
    summary_metrics,
    top_restaurants,
)
from src.data_cleaning import clean_restaurants, load_csv  # noqa: E402
from src.ml_model import model_coefficients, train_and_evaluate_models  # noqa: E402


st.set_page_config(
    page_title="RestroPulse | Restaurant Market Intelligence",
    page_icon="🍽️",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
      [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f5f9ff 0%, #edf4ff 100%);
        border: 1px solid #dce8f5; padding: 1rem; border-radius: 0.8rem;
      }
      [data-testid="stMetricLabel"] {color: #52677e;}
      h1, h2, h3 {color: #15344f;}
      .stCaption {color: #536b82;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_default_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    source = PROJECT_ROOT / "data" / "cleaned.csv"
    if not source.exists():
        raise FileNotFoundError(f"Default dataset not found: {source}")
    return load_csv(source)


@st.cache_resource(show_spinner="Training and evaluating the models on a holdout set…")
def _cached_model_run(model_frame: pd.DataFrame) -> dict:
    return train_and_evaluate_models(model_frame)


def _filter_restaurants(
    restaurants: pd.DataFrame,
    selected_location: str,
    selected_categories: list[str],
    search_text: str,
) -> pd.DataFrame:
    result = restaurants
    if selected_location != "All cities":
        result = result[result["location"].eq(selected_location)]
    if selected_categories:
        selected = {category.casefold() for category in selected_categories}
        result = result[
            result["categories_clean"].map(
                lambda categories: any(category.casefold() in selected for category in categories)
            )
        ]
    if search_text.strip():
        query = search_text.strip().casefold()
        result = result[
            result["name"].str.casefold().str.contains(query, na=False)
            | result["business_id"].str.casefold().str.contains(query, na=False)
        ]
    return result.copy()


def _restaurant_table(restaurants: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "name",
        "location",
        "categories_display",
        "stars",
        "review_count",
        "business_id",
    ]
    return restaurants[columns].rename(
        columns={
            "name": "Restaurant",
            "location": "City / region",
            "categories_display": "Categories",
            "stars": "Rating",
            "review_count": "Reviews",
            "business_id": "Business ID",
        }
    )


def main() -> None:
    st.title("RestroPulse")
    st.caption("Restaurant Trends & Competition Analysis")
    st.write(
        "Explore restaurant supply, customer review activity, ratings, and relative "
        "competition across the cities in your dataset."
    )

    with st.sidebar:
        st.header("Data and filters")
        uploaded = st.file_uploader("Upload a restaurant CSV", type=["csv"], key="dataset_upload")

    try:
        if uploaded is None:
            restaurants, quality = _load_default_dataset()
            data_source_label = "Included Yelp business dataset"
        else:
            restaurants, quality = load_csv(uploaded)
            data_source_label = uploaded.name
    except (ValueError, FileNotFoundError, TypeError) as exc:
        st.error(str(exc))
        st.info(
            "Include a restaurant rating column named stars, rating, or average_rating. "
            "Restaurant name, city, state, categories, and review count are optional."
        )
        st.stop()

    st.caption(
        f"Source: {data_source_label} · {quality['retained_rows']:,} usable restaurants"
    )
    with st.expander("Data quality summary"):
        quality_columns = st.columns(4)
        quality_columns[0].metric("Input rows", f"{quality['input_rows']:,}")
        quality_columns[1].metric("Usable rows", f"{quality['retained_rows']:,}")
        quality_columns[2].metric("Duplicates removed", f"{quality['duplicate_ids_removed']:,}")
        quality_columns[3].metric(
            "Invalid / missing ratings",
            f"{quality['invalid_ratings_removed'] + quality['missing_ratings_removed']:,}",
        )

    with st.sidebar:
        locations = ["All cities", *sorted(restaurants["location"].dropna().unique())]
        selected_location = st.selectbox("City / region", locations, key="city_filter")

        category_options = sorted(
            {
                category
                for categories in restaurants["categories_clean"]
                for category in categories
                if category.casefold() not in {"restaurants", "food"}
            },
            key=str.casefold,
        )
        selected_categories = st.multiselect(
            "Restaurant categories",
            options=category_options,
            key="category_filter",
            help="Selecting several categories includes restaurants matching any selected category.",
        )
        search_text = st.text_input("Search restaurant or business ID", key="restaurant_search")
        minimum_reviews = st.slider(
            "Minimum reviews for top / low rating lists",
            min_value=0,
            max_value=100,
            value=5,
            key="minimum_reviews_filter",
        )
        minimum_category_restaurants = st.slider(
            "Minimum restaurants per category",
            min_value=1,
            max_value=25,
            value=2,
            key="minimum_category_count_filter",
            help="Hides very sparse categories from competition and opportunity rankings.",
        )

    filtered = _filter_restaurants(
        restaurants,
        selected_location,
        selected_categories,
        search_text,
    )
    if filtered.empty:
        st.warning("No restaurants match these filters. Clear a filter or try another search.")
        st.stop()

    metrics = summary_metrics(filtered)
    kpis = st.columns(4)
    kpis[0].metric("Restaurants", f"{metrics['total_restaurants']:,}")
    kpis[1].metric("Average rating", f"{metrics['average_rating']:.2f} / 5")
    kpis[2].metric("Customer reviews", f"{metrics['total_reviews']:,}")
    kpis[3].metric("Cities / regions", f"{metrics['number_of_cities']:,}")

    overview_tab, competition_tab, restaurants_tab, ml_tab = st.tabs(
        ["Overview", "Competition & opportunity", "Restaurants", "ML lab"]
    )

    with overview_tab:
        left, right = st.columns(2)
        city_data = city_summary(filtered).head(15).sort_values("restaurant_count")
        with left:
            st.subheader("Restaurant supply by city")
            if len(city_data) == 1:
                st.info(f"{int(city_data.iloc[0]['restaurant_count']):,} restaurants in {city_data.iloc[0]['location']}.")
            else:
                fig = px.bar(
                    city_data,
                    x="restaurant_count",
                    y="location",
                    orientation="h",
                    color="average_rating",
                    color_continuous_scale="Teal",
                    labels={
                        "restaurant_count": "Restaurants",
                        "location": "City / region",
                        "average_rating": "Average rating",
                    },
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar_title="Rating")
                st.plotly_chart(fig, use_container_width=True)
        with right:
            st.subheader("Restaurant rating distribution")
            rating_fig = px.histogram(
                filtered,
                x="stars",
                nbins=9,
                range_x=[1, 5],
                labels={"stars": "Restaurant rating", "count": "Restaurants"},
                color_discrete_sequence=["#287a78"],
            )
            rating_fig.update_layout(bargap=0.12, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(rating_fig, use_container_width=True)

        st.subheader("Most common categories")
        categories = category_summary(filtered, minimum_restaurants=minimum_category_restaurants)
        if categories.empty:
            st.info("No categories meet the selected minimum restaurant count.")
        else:
            chart_data = categories.nlargest(15, "restaurant_count").sort_values("restaurant_count")
            fig = px.bar(
                chart_data,
                x="restaurant_count",
                y="category",
                orientation="h",
                color="average_rating",
                color_continuous_scale="Teal",
                labels={
                    "restaurant_count": "Restaurants tagged with category",
                    "category": "Category",
                    "average_rating": "Average rating",
                },
            )
            fig.update_layout(margin=dict(l=0, r=0, t=5, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with competition_tab:
        st.subheader("Competition is relative to the selected market")
        st.caption(
            "The score combines category supply, review activity per restaurant, and ratings. "
            "The opportunity index combines lower observed supply with reviews per existing venue; "
            "it is a proxy, not a forecast of demand or business success."
        )
        categories = category_summary(
            filtered,
            minimum_restaurants=minimum_category_restaurants,
        )
        if categories.empty:
            st.info("No categories meet the selected minimum restaurant count.")
        else:
            comp_col, opp_col = st.columns(2)
            with comp_col:
                st.markdown("#### Highest relative competition")
                comp_data = categories.nlargest(12, "competition_score").sort_values("competition_score")
                comp_fig = px.bar(
                    comp_data,
                    x="competition_score",
                    y="category",
                    orientation="h",
                    color="restaurant_count",
                    color_continuous_scale="Blues",
                    labels={
                        "competition_score": "Competition score (0–100)",
                        "category": "Category",
                        "restaurant_count": "Restaurants",
                    },
                )
                comp_fig.update_layout(margin=dict(l=0, r=0, t=5, b=0))
                st.plotly_chart(comp_fig, use_container_width=True)
            with opp_col:
                st.markdown("#### Highest opportunity proxy")
                opp_data = categories.nlargest(12, "opportunity_index").sort_values("opportunity_index")
                opp_fig = px.bar(
                    opp_data,
                    x="opportunity_index",
                    y="category",
                    orientation="h",
                    color="reviews_per_restaurant",
                    color_continuous_scale="Teal",
                    labels={
                        "opportunity_index": "Opportunity proxy (0–100)",
                        "category": "Category",
                        "reviews_per_restaurant": "Reviews per restaurant",
                    },
                )
                opp_fig.update_layout(margin=dict(l=0, r=0, t=5, b=0))
                st.plotly_chart(opp_fig, use_container_width=True)

            scatter = px.scatter(
                categories,
                x="restaurant_count",
                y="reviews_per_restaurant",
                size="total_reviews",
                color="competition_score",
                hover_name="category",
                hover_data={
                    "average_rating": ":.2f",
                    "opportunity_index": ":.1f",
                    "restaurant_count": True,
                    "reviews_per_restaurant": ":.1f",
                },
                color_continuous_scale="Teal",
                labels={
                    "restaurant_count": "Observed restaurant supply",
                    "reviews_per_restaurant": "Reviews per restaurant",
                    "competition_score": "Competition score",
                },
                title="Category supply and review activity",
            )
            st.plotly_chart(scatter, use_container_width=True)
            st.dataframe(
                categories.sort_values("opportunity_index", ascending=False),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "restaurant_share_pct": st.column_config.NumberColumn(format="%.1f%%"),
                    "review_share_pct": st.column_config.NumberColumn(format="%.1f%%"),
                    "average_rating": st.column_config.NumberColumn(format="%.2f"),
                    "review_weighted_rating": st.column_config.NumberColumn(format="%.2f"),
                    "competition_score": st.column_config.ProgressColumn(
                        "Competition", min_value=0, max_value=100, format="%.1f"
                    ),
                    "opportunity_index": st.column_config.ProgressColumn(
                        "Opportunity proxy", min_value=0, max_value=100, format="%.1f"
                    ),
                },
            )

            report = build_market_report(
                filtered,
                selected_location=None if selected_location == "All cities" else selected_location,
                minimum_category_restaurants=minimum_category_restaurants,
            )
            st.download_button(
                "Download market opportunity report",
                data=report,
                file_name="restropulse_market_report.md",
                mime="text/markdown",
            )
            st.download_button(
                "Download category analysis CSV",
                data=categories.to_csv(index=False).encode("utf-8"),
                file_name="restropulse_category_analysis.csv",
                mime="text/csv",
            )

    with restaurants_tab:
        st.subheader("Restaurant finder")
        st.caption("Top and lowest ratings are restricted by the review threshold in the sidebar.")
        high, low = st.columns(2)
        with high:
            st.markdown("#### Highest rated")
            high_rated = top_restaurants(
                filtered,
                minimum_reviews=minimum_reviews,
                ascending=False,
            )
            st.dataframe(_restaurant_table(high_rated), use_container_width=True, hide_index=True)
        with low:
            st.markdown("#### Lowest rated")
            low_rated = top_restaurants(
                filtered,
                minimum_reviews=minimum_reviews,
                ascending=True,
            )
            st.dataframe(_restaurant_table(low_rated), use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered restaurants CSV",
            data=_restaurant_table(filtered).to_csv(index=False).encode("utf-8"),
            file_name="restropulse_filtered_restaurants.csv",
            mime="text/csv",
        )

    with ml_tab:
        st.subheader("Machine learning experiments")
        st.warning(
            "Ratings are historical Yelp observations. The regression is an experiment using "
            "review count, category count, city, and region; it is not a forecast of future ratings."
        )
        if len(restaurants) < 20:
            st.info("At least 20 valid restaurants are needed to train and hold out a test set.")
        else:
            model_frame = restaurants[
                ["business_id", "city", "state", "stars", "review_count", "category_count"]
            ].copy()
            try:
                with st.spinner("Training models and scoring the holdout set…"):
                    result = _cached_model_run(model_frame)
                model_metrics = result["metrics"]
                metric_cols = st.columns(5)
                metric_cols[0].metric("Holdout MAE", f"{model_metrics['mae']:.3f} stars")
                metric_cols[1].metric("Holdout RMSE", f"{model_metrics['rmse']:.3f} stars")
                metric_cols[2].metric("Holdout R²", f"{model_metrics['r2']:.3f}")
                metric_cols[3].metric("Median baseline MAE", f"{model_metrics['baseline_mae']:.3f}")
                metric_cols[4].metric(
                    "MAE change vs baseline",
                    f"{model_metrics['mae_improvement_vs_baseline_pct']:+.1f}%",
                )
                st.caption(
                    f"Random 80/20 train/test split · {model_metrics['train_rows']:,} train rows · "
                    f"{model_metrics['test_rows']:,} holdout rows · metrics do not establish future performance."
                )

                pred = result["predictions"]
                prediction_fig = px.scatter(
                    pred,
                    x="actual_stars",
                    y="predicted_stars",
                    opacity=0.55,
                    labels={
                        "actual_stars": "Actual rating",
                        "predicted_stars": "Predicted rating",
                    },
                    title="Predicted vs actual ratings on holdout rows",
                )
                prediction_fig.add_shape(
                    type="line",
                    x0=1,
                    y0=1,
                    x1=5,
                    y1=5,
                    line=dict(color="#d04b45", dash="dash"),
                )
                prediction_fig.update_xaxes(range=[1, 5])
                prediction_fig.update_yaxes(range=[1, 5])
                st.plotly_chart(prediction_fig, use_container_width=True)

                segmented = result["segmented_restaurants"]
                filtered_ids = set(filtered["business_id"])
                visible_segments = segmented[segmented["business_id"].isin(filtered_ids)]
                visible_segments = visible_segments.assign(
                    log_review_count=np.log1p(visible_segments["review_count"])
                )
                left, right = st.columns(2)
                with left:
                    st.markdown("#### K-Means restaurant segments")
                    cluster_fig = px.scatter(
                        visible_segments,
                        x="log_review_count",
                        y="stars",
                        color=visible_segments["cluster"].astype(str),
                        size="category_count",
                        hover_data=["city", "state", "category_count"],
                        labels={
                            "log_review_count": "Log reviews (log1p)",
                            "stars": "Rating",
                            "color": "Cluster",
                        },
                    )
                    st.plotly_chart(cluster_fig, use_container_width=True)
                with right:
                    st.markdown("#### Cluster profiles")
                    profiles = (
                        visible_segments.groupby("cluster")
                        .agg(
                            restaurants=("business_id", "nunique"),
                            average_rating=("stars", "mean"),
                            median_reviews=("review_count", "median"),
                            average_categories=("category_count", "mean"),
                        )
                        .reset_index()
                    )
                    st.dataframe(profiles, use_container_width=True, hide_index=True)

                st.markdown("#### Strongest fitted associations")
                st.caption("Linear coefficients describe associations in the fitted dataset, not causal effects.")
                st.dataframe(
                    model_coefficients(result["rating_model"]).head(12),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "coefficient": st.column_config.NumberColumn(format="%.4f"),
                        "absolute_effect": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
                st.caption(
                    "Clustering silhouette score: "
                    + (
                        f"{model_metrics['silhouette_score']:.3f}"
                        if pd.notna(model_metrics["silhouette_score"])
                        else "not available for this data"
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"Could not train the models on this dataset: {exc}")

    st.divider()
    st.caption(
        "Data coverage and review behavior reflect the provided Yelp business snapshot. "
        "Category tags can overlap, and opportunity scores are descriptive screening tools."
    )


if __name__ == "__main__":
    main()
