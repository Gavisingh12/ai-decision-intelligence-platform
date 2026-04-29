"""Classification and recommendation training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.db.models import ClassificationRun, Dataset, DatasetRecord
from backend.schemas.classification import ClassificationPredictRequest, ClassificationTrainRequest
from backend.services.persistence import hub_persistence_service
from backend.services.storage import image_to_data_url


def _clean_json(value: Any) -> Any:
    """Convert numpy and pandas values to JSON-compatible Python values."""

    if isinstance(value, dict):
        return {key: _clean_json(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ClassificationService:
    """Train recommendation/classification models on structured datasets."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def train(
        self,
        *,
        db: Session,
        request: ClassificationTrainRequest,
    ) -> tuple[ClassificationRun, dict[str, str], dict[str, Any]]:
        """Train a classification model and persist artifacts."""

        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {request.dataset_id} was not found.")

        dataframe = self._load_dataset_frame(db=db, dataset_id=request.dataset_id)
        target_column = request.target_column
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

        if dataframe[target_column].nunique(dropna=True) < 2:
            raise ValueError("Classification requires at least two target classes.")

        working_frame = dataframe.dropna(subset=[target_column]).copy()
        self._validate_target(working_frame=working_frame, target_column=target_column)
        feature_columns = [column for column in working_frame.columns if column != target_column]
        if not feature_columns:
            raise ValueError("No feature columns remain after excluding the target column.")

        X = working_frame[feature_columns].copy()
        y_raw = working_frame[target_column].astype(str)

        numeric_columns = [
            column
            for column in feature_columns
            if pd.api.types.is_numeric_dtype(X[column])
        ]
        categorical_columns = [column for column in feature_columns if column not in numeric_columns]

        X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors="coerce")
        for column in categorical_columns:
            X[column] = X[column].fillna("missing").astype(str)

        # Fill numeric gaps with median values to avoid brittle training failures.
        for column in numeric_columns:
            median_value = X[column].median()
            X[column] = X[column].fillna(float(median_value) if not np.isnan(median_value) else 0.0)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        classes = label_encoder.classes_.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=request.test_size,
            random_state=42,
            stratify=y,
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", numeric_columns),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ],
            remainder="drop",
        )

        model_kwargs = {
            "objective": "multi:softprob" if len(classes) > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if len(classes) > 2 else "logloss",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
        }
        if len(classes) > 2:
            model_kwargs["num_class"] = len(classes)

        from xgboost import XGBClassifier

        if self.settings.free_mode:
            model_kwargs["n_estimators"] = 220

        model = XGBClassifier(**model_kwargs)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        metrics = self._evaluate(y_test=y_test, y_pred=y_pred)

        transformed_feature_names = list(
            pipeline.named_steps["preprocessor"].get_feature_names_out()
        )
        feature_importances = pipeline.named_steps["model"].feature_importances_

        artifact_dir = self.settings.model_dir / f"classification_run_{uuid4().hex}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.joblib"
        importance_chart_path = artifact_dir / "classification_importance.png"
        confusion_chart_path = artifact_dir / "classification_confusion.png"

        top_importances = self._top_feature_importances(
            feature_names=transformed_feature_names,
            feature_importances=feature_importances,
        )
        self._save_importance_plot(top_importances, importance_chart_path)
        self._save_confusion_plot(
            y_true=y_test,
            y_pred=y_pred,
            classes=classes,
            path=confusion_chart_path,
            max_classes=request.max_classes_for_report,
        )

        bundle = {
            "pipeline": pipeline,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
            "label_encoder": label_encoder,
        }
        joblib.dump(bundle, model_path)

        holdout_predictions = []
        decoded_pred = label_encoder.inverse_transform(y_pred)
        decoded_actual = label_encoder.inverse_transform(y_test)
        for idx in range(min(10, len(X_test))):
            class_probs = probabilities[idx]
            ranked_probs = sorted(
                [
                    {"label": classes[class_index], "probability": float(prob)}
                    for class_index, prob in enumerate(class_probs)
                ],
                key=lambda item: item["probability"],
                reverse=True,
            )[:3]
            holdout_predictions.append(
                {
                    "actual": decoded_actual[idx],
                    "predicted": decoded_pred[idx],
                    "top_probabilities": ranked_probs,
                }
            )

        metrics_json = _clean_json(
            {
                "evaluation": metrics,
                "classes": classes,
                "feature_columns": feature_columns,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "feature_hints": self._feature_hints(
                    frame=X,
                    feature_columns=feature_columns,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                ),
                "holdout_predictions": holdout_predictions,
                "artifacts": {
                    "importance_chart": str(importance_chart_path),
                    "confusion_chart": str(confusion_chart_path),
                },
                "insights": {
                    "top_features": top_importances,
                    "sample_prediction_count": len(holdout_predictions),
                },
            }
        )

        run = ClassificationRun(
            dataset_id=request.dataset_id,
            model_name="xgboost_classifier",
            target_column=target_column,
            artifact_path=str(model_path),
            metrics_json=metrics_json,
            status="completed",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        hub_persistence_service.sync_runtime_data(reason=f"classification training run {run.id}")

        return run, self._charts_from_run(run), metrics_json["insights"]

    def predict_with_run(
        self,
        *,
        db: Session,
        run_id: int,
        request: ClassificationPredictRequest,
    ) -> dict[str, Any]:
        """Predict class probabilities using a saved classification run."""

        run = db.query(ClassificationRun).filter(ClassificationRun.id == run_id).first()
        if not run:
            raise ValueError(f"Classification run {run_id} was not found.")

        bundle = joblib.load(run.artifact_path)
        pipeline: Pipeline = bundle["pipeline"]
        label_encoder: LabelEncoder = bundle["label_encoder"]
        feature_columns: list[str] = bundle["feature_columns"]
        numeric_columns: list[str] = bundle["numeric_columns"]
        categorical_columns: list[str] = bundle["categorical_columns"]

        row = {column: request.feature_values.get(column) for column in feature_columns}
        frame = pd.DataFrame([row])

        missing_columns = [column for column in feature_columns if column not in request.feature_values]
        if missing_columns:
            raise ValueError(
                f"Missing feature values for prediction: {', '.join(missing_columns)}"
            )

        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if frame[column].isna().any():
                raise ValueError(f"Feature '{column}' must be numeric for classification inference.")
        for column in categorical_columns:
            frame[column] = frame[column].fillna("missing").astype(str)

        probabilities = pipeline.predict_proba(frame)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = str(label_encoder.inverse_transform([predicted_index])[0])
        ranked_probs = sorted(
            [
                {"label": str(label_encoder.classes_[index]), "probability": float(prob)}
                for index, prob in enumerate(probabilities)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )

        return {
            "predicted_label": predicted_label,
            "probabilities": ranked_probs,
        }

    def serialize_run(self, run: ClassificationRun) -> tuple[dict[str, str], dict[str, Any]]:
        """Return frontend-ready charts and insights for a classification run."""

        return self._charts_from_run(run), run.metrics_json.get("insights", {})

    @staticmethod
    def _load_dataset_frame(*, db: Session, dataset_id: int) -> pd.DataFrame:
        """Rehydrate a dataset from stored records."""

        records = (
            db.query(DatasetRecord)
            .filter(DatasetRecord.dataset_id == dataset_id)
            .order_by(DatasetRecord.record_index.asc())
            .all()
        )
        if not records:
            raise ValueError("No structured records were found for the dataset.")

        return pd.DataFrame([record.payload for record in records])

    @staticmethod
    def _validate_target(*, working_frame: pd.DataFrame, target_column: str) -> None:
        """Reject obviously continuous numeric targets in the classification workflow."""

        target_series = working_frame[target_column]
        unique_count = int(target_series.nunique(dropna=True))
        row_count = int(len(target_series))
        if (
            pd.api.types.is_numeric_dtype(target_series)
            and unique_count > max(40, int(row_count * 0.2))
        ):
            raise ValueError(
                f"Target column '{target_column}' looks continuous rather than categorical. Use classification for labels such as crop names or classes, and keep numeric targets in forecasting/regression workflows."
            )

    @staticmethod
    def _evaluate(*, y_test: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute classification metrics."""

        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        }

    @staticmethod
    def _top_feature_importances(
        *,
        feature_names: list[str],
        feature_importances: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return top feature importances."""

        rankings = sorted(
            [
                {"feature": feature_names[index], "importance": float(value)}
                for index, value in enumerate(feature_importances)
            ],
            key=lambda item: item["importance"],
            reverse=True,
        )
        return rankings[:12]

    @staticmethod
    def _feature_hints(
        *,
        frame: pd.DataFrame,
        feature_columns: list[str],
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Build lightweight feature hints for frontend inference forms."""

        hints: dict[str, dict[str, Any]] = {}
        for column in feature_columns:
            series = frame[column]
            if column in numeric_columns:
                numeric_series = pd.to_numeric(series, errors="coerce").dropna()
                example = float(numeric_series.iloc[0]) if not numeric_series.empty else None
                median = float(numeric_series.median()) if not numeric_series.empty else None
                hints[column] = {
                    "type": "numeric",
                    "example": example,
                    "default": median,
                }
                continue

            categorical_series = series.dropna().astype(str)
            example = categorical_series.iloc[0] if not categorical_series.empty else None
            mode = categorical_series.mode()
            hints[column] = {
                "type": "categorical",
                "example": example,
                "default": str(mode.iloc[0]) if not mode.empty else example,
                "known_values": categorical_series.drop_duplicates().head(8).tolist(),
            }

        for column in categorical_columns:
            hints.setdefault(column, {"type": "categorical"})
        return hints

    @staticmethod
    def _save_importance_plot(top_importances: list[dict[str, Any]], path: Path) -> None:
        """Persist a feature-importance bar chart."""

        frame = pd.DataFrame(top_importances).sort_values("importance")
        plt.figure(figsize=(10, 5))
        plt.barh(frame["feature"], frame["importance"], color="#2563eb")
        plt.title("Classification Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    @staticmethod
    def _save_confusion_plot(
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        classes: list[str],
        path: Path,
        max_classes: int,
    ) -> None:
        """Persist a confusion matrix heatmap for the top classes."""

        labels = list(range(min(len(classes), max_classes)))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap="Blues")
        plt.title("Holdout Confusion Matrix")
        plt.colorbar()
        tick_labels = classes[: len(labels)]
        plt.xticks(range(len(tick_labels)), tick_labels, rotation=45, ha="right")
        plt.yticks(range(len(tick_labels)), tick_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

    @staticmethod
    def _charts_from_run(run: ClassificationRun) -> dict[str, str]:
        """Load saved classification artifacts as data URLs."""

        artifacts = run.metrics_json.get("artifacts", {})
        return {
            key: image_to_data_url(path)
            for key, path in artifacts.items()
            if str(path).endswith(".png")
        }


classification_service = ClassificationService()
