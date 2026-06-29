"""Generate README-ready model result tables.

Usage:
    python -m src.docs.generate_tables
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def load_metrics():
    rows = []

    logreg_path = MODELS_DIR / "logreg.metrics.json"
    xgb_path = MODELS_DIR / "xgb.metrics.json"
    tuned_path = REPORTS_DIR / "xgb_tuned_results.json"

    if logreg_path.exists():
        data = json.loads(logreg_path.read_text())
        rows.append({
            "Model": "Logistic Regression",
            "Accuracy": data["accuracy"],
            "AUC": data["auc"],
            "Log Loss": data["log_loss"],
            "Brier": data["brier_score"],
        })

    if xgb_path.exists():
        data = json.loads(xgb_path.read_text())
        rows.append({
            "Model": "XGBoost",
            "Accuracy": data["accuracy"],
            "AUC": data["auc"],
            "Log Loss": data["log_loss"],
            "Brier": data["brier_score"],
        })

    if tuned_path.exists():
        data = json.loads(tuned_path.read_text())["test_metrics"]
        rows.append({
            "Model": "Tuned XGBoost",
            "Accuracy": data["accuracy"],
            "AUC": data["auc"],
            "Log Loss": data["log_loss"],
            "Brier": data["brier_score"],
        })

    return pd.DataFrame(rows)


def save_markdown_table(df: pd.DataFrame):
    out = DOCS_DIR / "model_comparison.md"

    formatted = df.copy()
    for col in ["Accuracy", "AUC", "Log Loss", "Brier"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")

    out.write_text(formatted.to_markdown(index=False))
    print(f"Saved {out}")


def save_table_image(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 2.8))
    fig.patch.set_facecolor("#101820")
    ax.axis("off")

    formatted = df.copy()
    for col in ["Accuracy", "AUC", "Log Loss", "Brier"]:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")

    table = ax.table(
        cellText=formatted.values,
        colLabels=formatted.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#F2AA4C")
        if row == 0:
            cell.set_facecolor("#1B2630")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#101820")
            cell.set_text_props(color="white")

    out = DOCS_DIR / "model_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved {out}")


def save_summary_json(df: pd.DataFrame):
    tuned = df[df["Model"] == "Tuned XGBoost"].iloc[0].to_dict()
    summary = {
        "project": "NBA Expected Field Goal Model",
        "n_shots": 654092,
        "final_model": "Tuned XGBoost",
        "final_metrics": tuned,
        "train_seasons": ["2022-23", "2023-24"],
        "test_season": "2024-25",
    }

    out = DOCS_DIR / "project_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metrics()
    if df.empty:
        raise ValueError("No model metric files found.")

    save_markdown_table(df)
    save_table_image(df)
    save_summary_json(df)


if __name__ == "__main__":
    main()
