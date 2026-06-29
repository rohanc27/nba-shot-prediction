"""Freeze the final production model.

Copies the current tuned XGBoost model to models/xgb_final.joblib.

Usage:
    python -m src.models.freeze_final_model
"""

from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def main() -> None:
    source = MODELS_DIR / "xgb_tuned.joblib"
    target = MODELS_DIR / "xgb_final.joblib"

    if not source.exists():
        raise FileNotFoundError(f"Missing tuned model: {source}")

    shutil.copyfile(source, target)
    print(f"Frozen final model written to {target}")


if __name__ == "__main__":
    main()
