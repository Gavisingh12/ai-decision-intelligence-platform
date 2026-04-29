"""SHAP explainability helpers for tree-based models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class SHAPService:
    """Generate global and local SHAP insights."""

    def generate_artifacts(
        self,
        *,
        model,
        X_reference: pd.DataFrame,
        X_explain: pd.DataFrame,
        artifact_dir: Path,
    ) -> dict:
        """Create summary plots and return structured importance insights."""

        if X_reference.empty or X_explain.empty:
            return {"paths": {}, "insights": {}}

        import shap

        explainer = shap.TreeExplainer(model)
        reference_values = explainer.shap_values(X_reference)
        explain_values = explainer.shap_values(X_explain)

        if isinstance(reference_values, list):
            reference_values = reference_values[0]
        if isinstance(explain_values, list):
            explain_values = explain_values[0]

        global_importance = (
            pd.Series(np.abs(reference_values).mean(axis=0), index=X_reference.columns)
            .sort_values(ascending=False)
            .head(10)
        )
        local_row = pd.Series(explain_values[0], index=X_explain.columns)
        local_importance = local_row.reindex(local_row.abs().sort_values(ascending=False).index).head(10)

        global_path = artifact_dir / "shap_global.png"
        local_path = artifact_dir / "shap_local.png"

        self._save_bar_chart(
            series=global_importance.sort_values(),
            path=global_path,
            title="Global SHAP Feature Importance",
            xlabel="Mean |SHAP value|",
            color="#2563eb",
        )
        self._save_bar_chart(
            series=local_importance.sort_values(),
            path=local_path,
            title="Local SHAP Explanation (Latest Sample)",
            xlabel="Contribution to prediction",
            color="#0f766e",
        )

        return {
            "paths": {
                "global_importance": str(global_path),
                "local_explanation": str(local_path),
            },
            "insights": {
                "global_top_features": [
                    {"feature": feature, "importance": float(value)}
                    for feature, value in global_importance.items()
                ],
                "local_top_contributors": [
                    {"feature": feature, "contribution": float(value)}
                    for feature, value in local_importance.items()
                ],
            },
        }

    @staticmethod
    def _save_bar_chart(
        *,
        series: pd.Series,
        path: Path,
        title: str,
        xlabel: str,
        color: str,
    ) -> None:
        """Render a horizontal bar chart to disk."""

        plt.figure(figsize=(10, 5))
        series.plot(kind="barh", color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()


shap_service = SHAPService()
