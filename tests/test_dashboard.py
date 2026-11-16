from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_renders_and_restaurant_search_filters_rows() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=180)

    assert len(app.exception) == 0
    assert len(app.metric) >= 4
    assert any("RestroPulse" in title.value for title in app.title)
    assert len(app.file_uploader) == 1
    assert len(app.multiselect) == 1
    assert len(app.download_button) >= 3

    app.get_by_key("restaurant_search").set_value("restropulse-no-match")
    app.run(timeout=180)

    assert len(app.exception) == 0
    assert any("No restaurants match" in warning.value for warning in app.warning)

    app.get_by_key("restaurant_search").set_value("taco")
    app.run(timeout=180)
    assert len(app.exception) == 0
    assert len(app.dataframe) > 0
