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
