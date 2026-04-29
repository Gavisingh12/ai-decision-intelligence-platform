import numpy as np
import pandas as pd
import pytest

from backend.services.classification import ClassificationService
from backend.services.forecasting import ForecastingService


def test_forecast_feature_frame_excludes_duplicate_trend_index():
    service = ForecastingService()
    frame = pd.DataFrame(
        {
            "humidity": np.linspace(60, 75, 40),
            "temperature": np.linspace(22, 31, 40),
            "label": ["rice"] * 40,
        }
    )

    feature_frame, feature_columns, static_columns = service._build_feature_frame(
        dataframe=frame,
        target_column="humidity",
        time_column=None,
        lags=[1, 2, 3],
    )

    assert feature_frame.empty is False
    assert feature_columns.count("trend_index") == 1
    assert "trend_index" not in static_columns
    assert "label" not in feature_columns


def test_classification_rejects_continuous_numeric_targets():
    service = ClassificationService()
    frame = pd.DataFrame(
        {
            "humidity": np.linspace(60, 90, 120),
            "temperature": np.linspace(22, 34, 120),
            "rainfall": np.linspace(80, 200, 120),
        }
    )

    with pytest.raises(ValueError, match="looks continuous"):
        service._validate_target(working_frame=frame, target_column="humidity")


def test_classification_allows_low_cardinality_numeric_labels():
    service = ClassificationService()
    frame = pd.DataFrame(
        {
            "class_id": [0, 1, 2, 0, 1, 2] * 20,
            "temperature": np.linspace(22, 34, 120),
        }
    )

    service._validate_target(working_frame=frame, target_column="class_id")
