# RestroPulse

**Restaurant Trends & Competition Analysis** is a Python data application for exploring restaurant supply, ratings, review activity, and category competition across the places represented in the supplied data.

## Dashboard features

- Headline restaurant, rating, review, and city metrics.
- City and category filters plus restaurant and business ID search.
- CSV upload with column normalization and validation.
- City-wise restaurant counts and rating distributions.
- Highest and lowest rated restaurant lists with a minimum-review filter.
- Relative category competition scores and an opportunity screening proxy.
- Interactive Plotly charts and downloadable restaurant, category, and market report files.
- K-Means restaurant segments and a linear regression rating experiment with holdout evaluation.

## Quick start

Python 3.14 is the version used to build and run this project. In Windows PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`.

For notebooks and tests, install the development tools as well:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the analysis notebook:

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/Analysis.ipynb
```

Run the data validation and model tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Train, evaluate, and save the rating and clustering pipelines as `models/model.pkl`:

```powershell
.\.venv\Scripts\python.exe -m src.ml_model
```

## Data and cleaning

The included `data/cleaned.csv` contains 36,889 restaurant records from the original Yelp business dataset used by the source project. The original raw Yelp JSON Lines file is not included. To rebuild the cleaned table, place `yelp_academic_dataset_business.json` in `data/` and run `notebooks/Clean.ipynb`.

CSV uploads require a rating field (`stars`, `rating`, or `average_rating`). Restaurant name, city, region, categories, review count, and business ID are optional; absent optional fields receive safe defaults. The cleaner:

- parses category arrays, JSON lists, Python literal lists, and delimited category strings;
- normalizes city and state/province spelling without treating a state as a city;
- drops missing or out-of-range ratings and duplicate business IDs;
- converts review counts to nonnegative integers and fills missing counts with zero;
- keeps the original category tags and excludes generic `Restaurants` and `Food` tags from category comparisons.

City labels use the observed city plus state or province (for example, `Phoenix, AZ`). This prevents unrelated cities in the same state from being grouped together.

## Competition and opportunity methodology

Scores are calculated within the selected city or filtered dataset and are relative rankings, not universal measures.

- **Competition score:** 45% category restaurant-count percentile, 35% reviews per restaurant percentile, and 20% review-weighted rating percentile.
- **Opportunity index:** 55% inverse restaurant-count percentile and 45% reviews per restaurant percentile.

Restaurants can have several category tags, so category counts overlap. Review activity only describes listed restaurants that received reviews. The opportunity index is a screening proxy; it does not measure unobserved customer demand, profitability, or likelihood of success.

## Machine learning

The dashboard and `src/ml_model.py` train two actual models:

- **Linear regression:** predicts observed star ratings from log review count, category count, city, and state/province. A reproducible random 80/20 train/test split reports MAE, RMSE, R?, and comparison with a train-median baseline. The final pipeline is fit on all supplied rows only after holdout evaluation.
- **K-Means:** groups restaurants using scaled rating, log review count, and category count. The dashboard reports cluster profiles and a silhouette score when the data supports it.

These experiments do not establish future rating accuracy: the dataset has no review dates or later outcomes. Coefficients are associations, not causal effects. The dashboard evaluates the complete loaded dataset before sidebar filters; filters only change the displayed market.

## Project structure

```text
app.py
requirements.txt
requirements-dev.txt
README.md
LICENSE.md
data/
  cleaned.csv
notebooks/
  Clean.ipynb
  Analysis.ipynb
src/
  data_cleaning.py
  analysis.py
  ml_model.py
models/
  model.pkl        # created by python -m src.ml_model
  metrics.json     # holdout results for that model bundle
assets/
  logo.svg
tests/
  test_data_pipeline.py
  test_dashboard.py
```

## Streamlit deployment

The public GitHub repository is ready for Community Cloud. In Streamlit Community Cloud, create an app with:

- Repository: `Priyansh-Chandravanshi/RestroPulse`
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.14` (Advanced settings)
- App URL: `restropulse` if available

The runtime `requirements.txt` contains only dashboard dependencies. The app reads the included `data/cleaned.csv` and trains its models at runtime. Community Cloud needs access to the public GitHub repository; deployment also requires signing in and connecting GitHub to the Streamlit workspace. See the [Community Cloud deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [GitHub connection guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account).
