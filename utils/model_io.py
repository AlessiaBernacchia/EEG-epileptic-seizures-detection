from pathlib import Path
import pickle
import re
from typing import Any, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "Models"


def _safe_filename(value: str) -> str:
    filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")
    return filename or "model"


def save_model(
    model: Any,
    filename: Optional[str] = None,
    models_dir: Union[str, Path] = DEFAULT_MODELS_DIR,
    overwrite: bool = True,
) -> Path:
    """
    Save any trained model object in the Models directory.

    The full wrapper object is saved, so model-specific attributes such as
    scalers, selected features, thresholds, histories, and fitted estimators are
    kept together without changing the existing model classes.
    """
    models_path = Path(models_dir)
    if not models_path.is_absolute():
        models_path = PROJECT_ROOT / models_path
    models_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        model_name = getattr(model, "model_name", model.__class__.__name__)
        filename = _safe_filename(str(model_name))

    file_path = models_path / filename
    if file_path.suffix == "":
        file_path = file_path.with_suffix(".joblib")

    if file_path.exists() and not overwrite:
        raise FileExistsError(f"Model file already exists: {file_path}")

    try:
        import joblib

        joblib.dump(model, file_path)
    except ImportError:
        if file_path.suffix == ".joblib":
            file_path = file_path.with_suffix(".pkl")
        with file_path.open("wb") as f:
            pickle.dump(model, f)

    return file_path

