"""Decision assistant orchestration service."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from sqlalchemy.orm import Session

from backend.db.models import ClassificationRun, Dataset, ForecastRun
from backend.rag.pipeline import get_rag_service
from backend.schemas.classification import ClassificationPredictRequest
from backend.services.classification import classification_service
from backend.services.llm import llm_service


class DecisionAssistantService:
    """Combine forecasts, classification signals, explainability, and retrieval into one answer."""

    def answer(
        self,
        *,
        db: Session,
        question: str,
        dataset_id: int | None = None,
        forecast_run_id: int | None = None,
        classification_run_id: int | None = None,
        top_k: int = 4,
        provider: str | None = None,
    ) -> tuple[
        str,
        str,
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any] | None,
        list[dict[str, Any]],
    ]:
        """Generate a decision-focused answer."""

        dataset = self._resolve_dataset(db=db, dataset_id=dataset_id)
        dataset_metadata = dataset.metadata_json if dataset and isinstance(dataset.metadata_json, dict) else {}
        classification_workflow = dataset_metadata.get("recommended_workflow") == "classification"
        question_wants_forecast = self._question_is_forecast_oriented(question)
        should_include_classification = bool(classification_run_id) or classification_workflow

        classification_run = (
            self._resolve_classification_run(
                db=db,
                dataset_id=dataset.id if dataset else dataset_id,
                classification_run_id=classification_run_id,
            )
            if should_include_classification
            else None
        )
        should_include_forecast = False
        if forecast_run_id:
            should_include_forecast = not should_include_classification or question_wants_forecast
        elif dataset_id:
            should_include_forecast = not classification_workflow or question_wants_forecast
        forecast_run = (
            self._resolve_forecast_run(
                db=db,
                dataset_id=dataset.id if dataset else dataset_id,
                forecast_run_id=forecast_run_id,
            )
            if should_include_forecast
            else None
        )

        forecast_summary = self._summarize_forecast_run(forecast_run) if forecast_run else None
        explainability_summary = forecast_run.metrics_json.get("shap") if forecast_run else None
        classification_summary = (
            self._summarize_classification_run(classification_run)
            if classification_run
            else None
        )

        if classification_run and classification_summary is not None:
            prediction = self._classification_prediction_from_question(
                db=db,
                run=classification_run,
                question=question,
            )
            if prediction:
                classification_summary["prediction"] = prediction

        rag_sources = get_rag_service().retrieve(db=db, question=question, top_k=top_k)
        retrieval_context = "\n".join(
            f"- {source.document_title}: {source.excerpt}" for source in rag_sources
        )

        prompt = (
            "You are an AI decision assistant. Synthesize forecasting metrics, classification outputs, "
            "SHAP explanations, and retrieved context into a practical answer.\n\n"
            f"Question: {question}\n\n"
            f"Dataset summary:\n{dataset.metadata_json if dataset else 'No dataset metadata was provided.'}\n\n"
            f"Forecast summary:\n{forecast_summary or 'No forecast run was provided.'}\n\n"
            f"Explainability summary:\n{explainability_summary or 'No SHAP insights available.'}\n\n"
            f"Classification summary:\n{classification_summary or 'No classification run was provided.'}\n\n"
            f"Retrieved context:\n{retrieval_context or 'No related documents retrieved.'}\n\n"
            "Answer in three short sections: What happened, Why it matters, Recommended next steps."
        )
        system_prompt = (
            "You are a senior decision intelligence analyst. Stay grounded in the provided evidence "
            "and call out uncertainty instead of over-claiming."
        )
        llm_result = llm_service.generate(prompt=prompt, system_prompt=system_prompt, provider=provider)
        answer = (
            self._build_local_fallback(
                question=question,
                dataset=dataset,
                forecast_summary=forecast_summary,
                explainability_summary=explainability_summary,
                classification_summary=classification_summary,
                sources=[source.model_dump() for source in rag_sources],
            )
            if llm_result.provider == "fallback"
            else llm_result.answer
        )

        return (
            answer,
            llm_result.provider,
            forecast_summary,
            explainability_summary,
            classification_summary,
            [source.model_dump() for source in rag_sources],
        )

    @staticmethod
    def _resolve_dataset(*, db: Session, dataset_id: int | None) -> Dataset | None:
        """Resolve an explicit dataset when provided."""

        if not dataset_id:
            return None
        return db.query(Dataset).filter(Dataset.id == dataset_id).first()

    @staticmethod
    def _question_is_forecast_oriented(question: str) -> bool:
        """Detect when the user is explicitly asking for forecast or trend details."""

        patterns = (
            r"\bforecast\b",
            r"\bfuture\b",
            r"\btrend\b",
            r"\bprojection\b",
            r"\bprojected\b",
            r"\bover time\b",
            r"\btime series\b",
            r"\bnext\s+\d+\b",
            r"\bnext\s+(day|days|week|weeks|month|months|quarter|quarters|season|seasons|year|years)\b",
            r"\bupward\b",
            r"\bdownward\b",
            r"\bflat\b",
        )
        lowered = question.lower()
        return any(re.search(pattern, lowered) for pattern in patterns)

    @staticmethod
    def _resolve_forecast_run(
        *,
        db: Session,
        dataset_id: int | None,
        forecast_run_id: int | None,
    ) -> ForecastRun | None:
        """Resolve an explicit or latest forecast run."""

        query = db.query(ForecastRun)
        if forecast_run_id:
            if dataset_id:
                return query.filter(
                    ForecastRun.id == forecast_run_id,
                    ForecastRun.dataset_id == dataset_id,
                ).first()
            return query.filter(ForecastRun.id == forecast_run_id).first()
        if dataset_id:
            return (
                query.filter(ForecastRun.dataset_id == dataset_id)
                .order_by(ForecastRun.created_at.desc())
                .first()
            )
        return None

    @staticmethod
    def _resolve_classification_run(
        *,
        db: Session,
        dataset_id: int | None,
        classification_run_id: int | None,
    ) -> ClassificationRun | None:
        """Resolve an explicit or latest classification run."""

        query = db.query(ClassificationRun)
        if classification_run_id:
            if dataset_id:
                return query.filter(
                    ClassificationRun.id == classification_run_id,
                    ClassificationRun.dataset_id == dataset_id,
                ).first()
            return query.filter(ClassificationRun.id == classification_run_id).first()
        if dataset_id:
            return (
                query.filter(ClassificationRun.dataset_id == dataset_id)
                .order_by(ClassificationRun.created_at.desc())
                .first()
            )
        return None

    @staticmethod
    def _summarize_forecast_run(run: ForecastRun) -> dict[str, Any]:
        """Build a compact run summary for prompting and API responses."""

        forecast = run.metrics_json.get("future_forecast", [])
        trend = "flat"
        if len(forecast) >= 2:
            start = forecast[0].get("prediction", 0)
            end = forecast[-1].get("prediction", 0)
            if end > start * 1.02:
                trend = "upward"
            elif end < start * 0.98:
                trend = "downward"

        return {
            "run_id": run.id,
            "model_name": run.model_name,
            "target_column": run.target_column,
            "time_column": run.time_column,
            "horizon": run.horizon,
            "evaluation": run.metrics_json.get("evaluation", {}),
            "trend": trend,
            "warnings": run.metrics_json.get("warnings", []),
            "latest_forecast": forecast[-3:],
        }

    @staticmethod
    def _summarize_classification_run(run: ClassificationRun) -> dict[str, Any]:
        """Build a compact classification summary for prompting and API responses."""

        insights = run.metrics_json.get("insights", {})
        return {
            "run_id": run.id,
            "model_name": run.model_name,
            "target_column": run.target_column,
            "evaluation": run.metrics_json.get("evaluation", {}),
            "classes": run.metrics_json.get("classes", []),
            "top_features": insights.get("top_features", []),
            "feature_columns": run.metrics_json.get("feature_columns", []),
        }

    def _classification_prediction_from_question(
        self,
        *,
        db: Session,
        run: ClassificationRun,
        question: str,
    ) -> dict[str, Any] | None:
        """Predict a label when the question includes usable feature assignments."""

        feature_values = self._extract_feature_values(
            question=question,
            feature_columns=run.metrics_json.get("feature_columns", []),
            numeric_columns=run.metrics_json.get("numeric_columns", []),
        )
        if not feature_values:
            return None

        required_features = set(run.metrics_json.get("feature_columns", []))
        if not required_features.issubset(feature_values.keys()):
            return {
                "status": "partial_input",
                "provided_features": feature_values,
                "missing_features": sorted(required_features - set(feature_values.keys())),
            }

        prediction = classification_service.predict_with_run(
            db=db,
            run_id=run.id,
            request=ClassificationPredictRequest(feature_values=feature_values),
        )
        top_probabilities = prediction.get("probabilities", [])[:3]
        return {
            "status": "predicted",
            "predicted_label": prediction["predicted_label"],
            "top_probabilities": top_probabilities,
            "provided_features": feature_values,
        }

    @staticmethod
    def _extract_feature_values(
        *,
        question: str,
        feature_columns: list[str],
        numeric_columns: list[str],
    ) -> dict[str, Any]:
        """Parse simple `feature=value` pairs from the user's question."""

        extracted: dict[str, Any] = {}
        numeric_set = {column.lower() for column in numeric_columns}
        for column in feature_columns:
            pattern = re.compile(rf"(?i)\b{re.escape(column)}\s*=\s*([a-z0-9._-]+)")
            match = pattern.search(question)
            if not match:
                continue
            raw_value = match.group(1).strip().rstrip(".,;)")
            if column.lower() in numeric_set:
                try:
                    extracted[column] = float(raw_value)
                except ValueError:
                    continue
            else:
                extracted[column] = raw_value
        return extracted

    @staticmethod
    def _build_local_fallback(
        *,
        question: str,
        dataset: Dataset | None,
        forecast_summary: dict[str, Any] | None,
        explainability_summary: dict[str, Any] | None,
        classification_summary: dict[str, Any] | None,
        sources: list[dict[str, Any]],
    ) -> str:
        """Return a useful local answer when no external LLM key is configured."""

        what_happened: list[str] = []
        why_it_matters: list[str] = []
        next_steps: list[str] = []

        prediction = classification_summary.get("prediction") if classification_summary else None
        if prediction and prediction.get("status") == "predicted":
            what_happened.append(
                f"Based on the latest classification run, the most suitable label is {prediction['predicted_label']}."
            )
            top_probabilities = prediction.get("top_probabilities", [])
            if top_probabilities:
                ranked = ", ".join(
                    f"{item['label']} ({item['probability']:.3f})" for item in top_probabilities
                )
                why_it_matters.append(f"Top model probabilities were: {ranked}.")
        elif prediction and prediction.get("status") == "partial_input":
            missing = ", ".join(prediction.get("missing_features", []))
            what_happened.append(
                "I found some feature values in your question, but not enough for a direct classification prediction."
            )
            why_it_matters.append(f"Missing values for: {missing}.")
        elif classification_summary:
            what_happened.append(
                "I used the latest recommendation model for this file to explain which inputs matter most."
            )

        if classification_summary:
            metrics = classification_summary.get("evaluation", {})
            accuracy = metrics.get("accuracy")
            if accuracy is not None:
                why_it_matters.append(
                    f"The latest classification run had about {accuracy:.3f} accuracy on holdout data."
                )
            top_features = classification_summary.get("top_features", [])[:3]
            if top_features:
                drivers = ", ".join(
                    f"{item['feature']} ({item['importance']:.3f})" for item in top_features
                )
                why_it_matters.append(f"Most influential model drivers were {drivers}.")
            next_steps.append(
                "Use the Classification view to compare the top predicted labels and validate them against local field conditions."
            )

        if forecast_summary:
            what_happened.append(
                f"The latest forecast for {forecast_summary['target_column']} shows a {forecast_summary['trend']} trend."
            )
            forecast_tail = DecisionAssistantService._format_forecast_tail(forecast_summary)
            if forecast_tail:
                tail_text = ", ".join(forecast_tail)
                why_it_matters.append(f"Recent forecast points were {tail_text}.")
            for warning in forecast_summary.get("warnings", []):
                next_steps.append(DecisionAssistantService._humanize_forecast_warning(warning))

        if explainability_summary:
            global_features = explainability_summary.get("global_top_features", [])[:3]
            if global_features:
                drivers = ", ".join(
                    f"{item['feature']} ({item['importance']:.3f})" for item in global_features
                )
                why_it_matters.append(f"SHAP highlights these global drivers: {drivers}.")

        if sources:
            source_titles = ", ".join(sorted({source["document_title"] for source in sources}))
            why_it_matters.append(f"Retrieved supporting documents: {source_titles}.")
        else:
            next_steps.append(
                "Upload PDF or text reports in Data Hub if you want document-grounded recommendations as well."
            )

        if dataset and dataset.metadata_json.get("recommended_workflow") == "classification":
            next_steps.append(
                "This file is better suited to recommendation questions, so crop suitability answers should come from the recommendation model first."
            )

        if not what_happened:
            what_happened.append(
                "I do not have enough model context yet to answer this directly, so the answer is limited."
            )
        if not why_it_matters:
            why_it_matters.append(
                "No forecast, classification, or retrieval evidence was available for a grounded explanation."
            )
        if not next_steps:
            next_steps.append(
                "Train a model or upload supporting reports so the assistant has stronger evidence to work with."
            )

        return (
            "What happened\n"
            + "\n".join(f"- {line}" for line in what_happened)
            + "\n\nWhy it matters\n"
            + "\n".join(f"- {line}" for line in why_it_matters)
            + "\n\nRecommended next steps\n"
            + "\n".join(f"- {line}" for line in next_steps)
        )

    @staticmethod
    def _format_forecast_tail(forecast_summary: dict[str, Any]) -> list[str]:
        """Format forecast points in a user-friendly way."""

        time_column = forecast_summary.get("time_column")
        formatted: list[str] = []
        for position, row in enumerate(forecast_summary.get("latest_forecast", []), start=1):
            label = DecisionAssistantService._format_forecast_label(
                timestamp=row.get("timestamp"),
                position=position,
                time_column=time_column,
            )
            prediction = row.get("prediction")
            prediction_text = (
                f"{float(prediction):.2f}"
                if isinstance(prediction, (int, float))
                else str(prediction)
            )
            formatted.append(f"{label}: {prediction_text}")
        return formatted

    @staticmethod
    def _format_forecast_label(
        *,
        timestamp: Any,
        position: int,
        time_column: str | None,
    ) -> str:
        """Format either a time value or a step counter for forecast points."""

        if not time_column:
            step_value = DecisionAssistantService._coerce_step_number(timestamp)
            return f"Step {step_value}" if step_value is not None else f"Step {position}"

        timestamp_text = DecisionAssistantService._clean_timestamp_text(timestamp)
        return timestamp_text or f"Point {position}"

    @staticmethod
    def _coerce_step_number(value: Any) -> int | None:
        """Convert a raw step value into an integer when possible."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric.is_integer():
            return int(numeric)
        return None

    @staticmethod
    def _clean_timestamp_text(value: Any) -> str | None:
        """Reduce timestamp formatting noise for fallback answers."""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d %H:%M")

        text = str(value).strip()
        if not text:
            return None

        match = re.match(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?$", text)
        if match:
            date_part, time_part = match.groups()
            return f"{date_part} {time_part}"

        return text

    @staticmethod
    def _humanize_forecast_warning(warning: str) -> str:
        """Rewrite technical forecast caveats into plainer language."""

        ignored_prefix = "Ignored unsupported categorical features for forecasting: "
        if warning.startswith(ignored_prefix):
            ignored_fields = warning.removeprefix(ignored_prefix)
            return (
                "This future estimate used number-based inputs only, so these non-number fields were skipped: "
                f"{ignored_fields}."
            )

        if warning.startswith("No time column was supplied."):
            return (
                "No date column was provided, so this estimate follows the current row order instead of calendar time."
            )

        return warning


decision_assistant_service = DecisionAssistantService()
