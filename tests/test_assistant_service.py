from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.services.assistant as assistant_module
from backend.db.base import Base
from backend.db.models import ClassificationRun, Dataset, ForecastRun
from backend.services.assistant import DecisionAssistantService


def test_extract_feature_values_from_question():
    values = DecisionAssistantService._extract_feature_values(
        question="Given N=90, P=40, K=40, temperature=25, humidity=80, ph=6.5, rainfall=200, which crop fits best?",
        feature_columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
        numeric_columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )

    assert values == {
        "N": 90.0,
        "P": 40.0,
        "K": 40.0,
        "temperature": 25.0,
        "humidity": 80.0,
        "ph": 6.5,
        "rainfall": 200.0,
    }


def test_local_fallback_uses_classification_prediction():
    answer = DecisionAssistantService._build_local_fallback(
        question="Which crop is most suitable?",
        dataset=None,
        forecast_summary=None,
        explainability_summary=None,
        classification_summary={
            "evaluation": {"accuracy": 0.98},
            "top_features": [
                {"feature": "humidity", "importance": 0.21},
                {"feature": "rainfall", "importance": 0.18},
            ],
            "prediction": {
                "status": "predicted",
                "predicted_label": "rice",
                "top_probabilities": [
                    {"label": "rice", "probability": 0.71},
                    {"label": "jute", "probability": 0.12},
                ],
            },
        },
        sources=[],
    )

    assert "most suitable label is rice" in answer
    assert "0.980 accuracy" in answer
    assert "humidity (0.210)" in answer


def test_local_fallback_formats_step_based_forecast_points_cleanly():
    answer = DecisionAssistantService._build_local_fallback(
        question="What does the temperature trend look like?",
        dataset=None,
        forecast_summary={
            "target_column": "temperature",
            "trend": "flat",
            "time_column": None,
            "warnings": [
                "Ignored unsupported categorical features for forecasting: label",
                "No time column was supplied. Forecasting uses row order as the sequence, so results depend on the current dataset ordering.",
            ],
            "latest_forecast": [{"timestamp": 99, "prediction": 27.23}],
        },
        explainability_summary=None,
        classification_summary=None,
        sources=[],
    )

    assert "Step 99: 27.23" in answer
    assert "1970-01-01" not in answer
    assert "row order instead of calendar time" in answer
    assert "number-based inputs only" in answer


def test_answer_ignores_stale_forecast_context_for_classification_questions(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    session = SessionLocal()

    crop_dataset = Dataset(
        name="Crop Recommendation",
        filename="crop.csv",
        file_path="crop.csv",
        row_count=2200,
        metadata_json={"recommended_workflow": "classification"},
    )
    weather_dataset = Dataset(
        name="Weather History",
        filename="weather.csv",
        file_path="weather.csv",
        row_count=365,
        metadata_json={"recommended_workflow": "forecasting"},
    )
    session.add_all([crop_dataset, weather_dataset])
    session.flush()

    classification_run = ClassificationRun(
        dataset_id=crop_dataset.id,
        target_column="label",
        artifact_path="classification.pkl",
        metrics_json={
            "evaluation": {"accuracy": 0.98},
            "insights": {"top_features": [{"feature": "humidity", "importance": 0.21}]},
            "feature_columns": [],
            "numeric_columns": [],
        },
    )
    forecast_run = ForecastRun(
        dataset_id=weather_dataset.id,
        target_column="temperature",
        time_column=None,
        horizon=14,
        artifact_path="forecast.pkl",
        metrics_json={
            "future_forecast": [{"timestamp": 99, "prediction": 27.23}],
            "warnings": [
                "No time column was supplied. Forecasting uses row order as the sequence, so results depend on the current dataset ordering."
            ],
        },
    )
    session.add_all([classification_run, forecast_run])
    session.commit()

    monkeypatch.setattr(
        assistant_module,
        "get_rag_service",
        lambda: SimpleNamespace(retrieve=lambda **_: []),
    )
    monkeypatch.setattr(
        assistant_module.llm_service,
        "generate",
        lambda **_: SimpleNamespace(provider="fallback", answer=""),
    )

    answer, provider, forecast_summary, _, classification_summary, sources = DecisionAssistantService().answer(
        db=session,
        question="Which crop is most suitable and why?",
        dataset_id=crop_dataset.id,
        forecast_run_id=forecast_run.id,
        classification_run_id=classification_run.id,
    )

    session.close()

    assert provider == "fallback"
    assert forecast_summary is None
    assert classification_summary is not None
    assert sources == []
    assert "most suitable label" not in answer.lower()
    assert "latest forecast" not in answer.lower()
    assert "0.980 accuracy" in answer
