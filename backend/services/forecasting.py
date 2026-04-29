"""Forecast training, persistence, and prediction utilities."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy.orm import Session

from backend.core.config import get_settings
from backend.db.models import Dataset, DatasetRecord, ForecastRun
from backend.explainability.shap_service import shap_service
from backend.schemas.forecast import ForecastTrainRequest
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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class ForecastingService:
    """Train XGBoost time-series models with lag features and SHAP insights."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def train(self, *, db: Session, request: ForecastTrainRequest) -> tuple[ForecastRun, dict[str, str], dict[str, Any]]:
        """Train a forecast model and persist its artifacts."""

        dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()
        if not dataset:
            raise ValueError(f"Dataset {request.dataset_id} was not found.")

        dataframe = self._load_dataset_frame(db=db, dataset_id=request.dataset_id)
        target_column = request.target_column
        if target_column not in dataframe.columns:
            raise ValueError(f"Target column '{target_column}' is not present in the dataset.")

        if not pd.api.types.is_numeric_dtype(dataframe[target_column]):
            raise ValueError(
                f"Target column '{target_column}' is not numeric. This looks like a classification or recommendation dataset; the current forecasting engine requires numeric targets."
            )

        time_column = request.time_column or dataset.time_column
        if time_column and time_column in dataframe.columns:
            dataframe[time_column] = pd.to_datetime(dataframe[time_column], errors="coerce")
            dataframe = dataframe.dropna(subset=[time_column]).sort_values(time_column).reset_index(drop=True)
        else:
            time_column = None
            dataframe = dataframe.reset_index(drop=True)

        dataframe[target_column] = pd.to_numeric(dataframe[target_column], errors="coerce")
        dataframe = dataframe.dropna(subset=[target_column]).reset_index(drop=True)
        if len(dataframe) < max(25, request.horizon + max(request.lags) + 5):
            raise ValueError("The dataset is too small for reliable forecasting after lag feature generation.")

        feature_frame, feature_columns, static_feature_columns = self._build_feature_frame(
            dataframe=dataframe,
            target_column=target_column,
            time_column=time_column,
            lags=request.lags,
        )
        if len(feature_frame) < 20:
            raise ValueError("Not enough rows remain after feature engineering to train the model.")

        split_index = max(int(len(feature_frame) * (1 - request.test_size)), len(feature_frame) - max(request.horizon, 3))
        split_index = min(max(split_index, 10), len(feature_frame) - 1)

        train_frame = feature_frame.iloc[:split_index].copy()
        test_frame = feature_frame.iloc[split_index:].copy()
        X_train = train_frame[feature_columns]
        y_train = train_frame[target_column]
        X_test = test_frame[feature_columns]
        y_test = test_frame[target_column]

        from xgboost import XGBRegressor

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=220 if self.settings.free_mode else 350,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train, y_train)

        test_predictions = model.predict(X_test)
        metrics = self._evaluate(y_test, test_predictions)
        frequency = self._infer_frequency(dataframe, time_column)
        future_frame, future_predictions = self._forecast_future(
            dataframe=dataframe,
            model=model,
            target_column=target_column,
            time_column=time_column,
            feature_columns=feature_columns,
            static_feature_columns=static_feature_columns,
            lags=request.lags,
            horizon=request.horizon,
            frequency=frequency,
        )

        artifact_dir = self.settings.model_dir / f"forecast_run_{uuid4().hex}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "model.joblib"
        chart_path = artifact_dir / "forecast.png"

        self._save_forecast_plot(
            dataframe=dataframe,
            target_column=target_column,
            time_column=time_column,
            test_frame=test_frame,
            test_predictions=test_predictions,
            future_frame=future_frame,
            future_predictions=future_predictions,
            chart_path=chart_path,
        )

        shap_payload = shap_service.generate_artifacts(
            model=model,
            X_reference=X_train.tail(min(self.settings.shap_reference_sample_size, len(X_train))),
            X_explain=X_test.tail(1) if not X_test.empty else X_train.tail(1),
            artifact_dir=artifact_dir,
        )

        bundle = {
            "model": model,
            "target_column": target_column,
            "time_column": time_column,
            "feature_columns": feature_columns,
            "static_feature_columns": static_feature_columns,
            "lags": request.lags,
            "frequency": frequency,
        }
        joblib.dump(bundle, model_path)

        metrics_json = _clean_json(
            {
                "evaluation": metrics,
                "holdout_predictions": [
                    {
                        "timestamp": feature_frame.iloc[index][time_column] if time_column else index,
                        "actual": y_test.iloc[offset],
                        "predicted": test_predictions[offset],
                    }
                    for offset, index in enumerate(test_frame.index.tolist())
                ],
                "future_forecast": future_frame.assign(prediction=future_predictions).to_dict(orient="records"),
                "feature_columns": feature_columns,
                "frequency": frequency,
                "warnings": self._forecast_warnings(
                    dataframe=dataframe,
                    target_column=target_column,
                    time_column=time_column,
                ),
                "shap": shap_payload["insights"],
                "artifacts": {
                    "forecast_chart": str(chart_path),
                    **shap_payload["paths"],
                },
            }
        )

        run = ForecastRun(
            dataset_id=request.dataset_id,
            model_name="xgboost",
            target_column=target_column,
            time_column=time_column,
            horizon=request.horizon,
            artifact_path=str(model_path),
            metrics_json=metrics_json,
            status="completed",
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        hub_persistence_service.sync_runtime_data(reason=f"forecast training run {run.id}")

        return run, self._charts_from_run(run), shap_payload["insights"]

    def predict_with_run(self, *, db: Session, run_id: int, horizon: int | None = None) -> dict[str, Any]:
        """Generate a fresh recursive forecast from a previously trained run."""

        run = db.query(ForecastRun).filter(ForecastRun.id == run_id).first()
        if not run:
            raise ValueError(f"Forecast run {run_id} was not found.")

        bundle = joblib.load(run.artifact_path)
        dataframe = self._load_dataset_frame(db=db, dataset_id=run.dataset_id)
        dataframe[bundle["target_column"]] = pd.to_numeric(dataframe[bundle["target_column"]], errors="coerce")
        if bundle["time_column"] and bundle["time_column"] in dataframe.columns:
            dataframe[bundle["time_column"]] = pd.to_datetime(dataframe[bundle["time_column"]], errors="coerce")
            dataframe = dataframe.dropna(subset=[bundle["time_column"]])
            dataframe = dataframe.sort_values(bundle["time_column"]).reset_index(drop=True)
        dataframe = dataframe.dropna(subset=[bundle["target_column"]]).reset_index(drop=True)

        future_frame, predictions = self._forecast_future(
            dataframe=dataframe,
            model=bundle["model"],
            target_column=bundle["target_column"],
            time_column=bundle["time_column"],
            feature_columns=bundle["feature_columns"],
            static_feature_columns=bundle["static_feature_columns"],
            lags=bundle["lags"],
            horizon=horizon or run.horizon,
            frequency=bundle["frequency"],
        )

        return {
            "run_id": run.id,
            "horizon": horizon or run.horizon,
            "forecast": _clean_json(
                future_frame.assign(prediction=predictions).to_dict(orient="records")
            ),
        }

    def serialize_run(self, run: ForecastRun) -> tuple[dict[str, str], dict[str, Any]]:
        """Return frontend-ready charts and explainability summaries for a run."""

        return self._charts_from_run(run), run.metrics_json.get("shap", {})

    def _load_dataset_frame(self, *, db: Session, dataset_id: int) -> pd.DataFrame:
        """Rehydrate a dataset from row-level records."""

        records = (
            db.query(DatasetRecord)
            .filter(DatasetRecord.dataset_id == dataset_id)
            .order_by(DatasetRecord.record_index.asc())
            .all()
        )
        if not records:
            raise ValueError("No structured records were found for the dataset.")

        return pd.DataFrame([record.payload for record in records])

    def _build_feature_frame(
        self,
        *,
        dataframe: pd.DataFrame,
        target_column: str,
        time_column: str | None,
        lags: list[int],
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Create lag, rolling, calendar, and covariate features."""

        frame = dataframe.copy()
        frame["trend_index"] = np.arange(len(frame))

        static_feature_columns: list[str] = []
        calendar_columns: list[str] = []
        lag_columns: list[str] = []
        numeric_feature_columns = [
            column
            for column in frame.columns
            if column not in {target_column, time_column, "trend_index"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]

        for column in numeric_feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            static_feature_columns.append(column)

        if time_column:
            frame["month"] = frame[time_column].dt.month
            frame["week"] = frame[time_column].dt.isocalendar().week.astype(int)
            frame["day"] = frame[time_column].dt.day
            frame["day_of_week"] = frame[time_column].dt.dayofweek
            frame["quarter"] = frame[time_column].dt.quarter
            calendar_columns = ["month", "week", "day", "day_of_week", "quarter"]

        for lag in lags:
            frame[f"lag_{lag}"] = frame[target_column].shift(lag)
            lag_columns.append(f"lag_{lag}")

        frame["rolling_mean_3"] = frame[target_column].shift(1).rolling(3).mean()
        frame["rolling_mean_7"] = frame[target_column].shift(1).rolling(7).mean()
        frame["rolling_std_7"] = frame[target_column].shift(1).rolling(7).std()

        frame = frame.dropna().reset_index(drop=True)
        feature_columns = (
            ["trend_index"]
            + static_feature_columns
            + calendar_columns
            + lag_columns
            + ["rolling_mean_3", "rolling_mean_7", "rolling_std_7"]
        )
        return frame, feature_columns, static_feature_columns

    @staticmethod
    def _forecast_warnings(
        *,
        dataframe: pd.DataFrame,
        target_column: str,
        time_column: str | None,
    ) -> list[str]:
        """Return user-facing forecast caveats."""

        warnings: list[str] = []
        ignored_categorical_features = [
            column
            for column in dataframe.columns
            if column not in {target_column, time_column}
            and not pd.api.types.is_numeric_dtype(dataframe[column])
        ]
        if ignored_categorical_features:
            warnings.append(
                "Ignored unsupported categorical features for forecasting: "
                + ", ".join(ignored_categorical_features)
            )
        if not time_column:
            warnings.append(
                "No time column was supplied. Forecasting uses row order as the sequence, so results depend on the current dataset ordering."
            )
        return warnings

    def _infer_frequency(self, dataframe: pd.DataFrame, time_column: str | None) -> str:
        """Infer a pandas-compatible frequency string from the dataset."""

        if not time_column:
            return "step"

        inferred = pd.infer_freq(dataframe[time_column])
        if inferred:
            return inferred

        deltas = dataframe[time_column].sort_values().diff().dropna()
        if deltas.empty:
            return "D"

        median_delta = deltas.median()
        if median_delta <= timedelta(hours=1):
            return "H"
        if median_delta <= timedelta(days=1):
            return "D"
        if median_delta <= timedelta(days=7):
            return "W"
        return "M"

    def _forecast_future(
        self,
        *,
        dataframe: pd.DataFrame,
        model,
        target_column: str,
        time_column: str | None,
        feature_columns: list[str],
        static_feature_columns: list[str],
        lags: list[int],
        horizon: int,
        frequency: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Generate recursive predictions for future periods."""

        history = dataframe[target_column].astype(float).tolist()
        static_defaults = {}
        for column in static_feature_columns:
            series = pd.to_numeric(dataframe[column], errors="coerce")
            static_defaults[column] = float(series.dropna().iloc[-1]) if not series.dropna().empty else 0.0

        if time_column:
            last_timestamp = pd.to_datetime(dataframe[time_column].iloc[-1])
            if frequency == "step":
                frequency = "D"
            future_timestamps = list(pd.date_range(start=last_timestamp, periods=horizon + 1, freq=frequency)[1:])
        else:
            future_timestamps = [None] * horizon

        rows: list[dict[str, Any]] = []
        predictions: list[float] = []
        for step in range(horizon):
            timestamp = future_timestamps[step]
            row: dict[str, Any] = {"trend_index": len(dataframe) + step}
            row.update(static_defaults)

            if time_column and timestamp is not None:
                row[time_column] = timestamp
                row["month"] = timestamp.month
                row["week"] = int(timestamp.isocalendar().week)
                row["day"] = timestamp.day
                row["day_of_week"] = timestamp.dayofweek
                row["quarter"] = timestamp.quarter

            for lag in lags:
                row[f"lag_{lag}"] = history[-lag]

            recent = history[-7:]
            row["rolling_mean_3"] = float(np.mean(history[-3:]))
            row["rolling_mean_7"] = float(np.mean(recent))
            row["rolling_std_7"] = float(np.std(recent))

            feature_row = pd.DataFrame([{column: row.get(column, 0.0) for column in feature_columns}])
            prediction = float(model.predict(feature_row)[0])
            history.append(prediction)
            predictions.append(prediction)
            rows.append(
                {
                    "timestamp": timestamp if timestamp is not None else len(dataframe) + step,
                }
            )

        future_frame = pd.DataFrame(rows)
        return future_frame, np.asarray(predictions)

    def _evaluate(self, actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
        """Compute regression metrics for holdout predictions."""

        non_zero = actual.replace(0, np.nan)
        mape = np.nanmean(np.abs((actual - predicted) / non_zero)) * 100
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else 0.0,
            "mape": float(mape) if not np.isnan(mape) else 0.0,
        }

    def _save_forecast_plot(
        self,
        *,
        dataframe: pd.DataFrame,
        target_column: str,
        time_column: str | None,
        test_frame: pd.DataFrame,
        test_predictions: np.ndarray,
        future_frame: pd.DataFrame,
        future_predictions: np.ndarray,
        chart_path: Path,
    ) -> None:
        """Save a forecast chart for the frontend."""

        plt.figure(figsize=(11, 5))
        x_actual = dataframe[time_column] if time_column else dataframe.index
        plt.plot(x_actual, dataframe[target_column], label="Actual", color="#0f172a")

        if not test_frame.empty:
            x_test = test_frame[time_column] if time_column else test_frame.index
            plt.plot(x_test, test_predictions, label="Holdout prediction", color="#2563eb")

        x_future = future_frame["timestamp"]
        plt.plot(x_future, future_predictions, label="Future forecast", color="#dc2626", linestyle="--")
        plt.title("Forecast vs Historical Actuals")
        plt.xlabel("Time")
        plt.ylabel(target_column)
        plt.legend()
        plt.tight_layout()
        plt.savefig(chart_path, dpi=160)
        plt.close()

    def _charts_from_run(self, run: ForecastRun) -> dict[str, str]:
        """Load run artifacts and return them as data URLs."""

        artifacts = run.metrics_json.get("artifacts", {})
        return {
            key: image_to_data_url(path)
            for key, path in artifacts.items()
            if path.endswith(".png")
        }


forecasting_service = ForecastingService()
