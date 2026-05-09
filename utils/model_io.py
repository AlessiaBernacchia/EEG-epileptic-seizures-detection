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


def load_model(path: Union[str, Path], map_location: str = "cpu") -> Any:
    path = Path(path)

    if path.suffix == ".joblib":
        import torch

        original_torch_load = torch.load

        def torch_load_with_map_location(*args, **kwargs):
            kwargs.setdefault("map_location", torch.device(map_location))
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_with_map_location

        try:
            import joblib
            model = joblib.load(path)
        finally:
            torch.load = original_torch_load

    else:
        with path.open("rb") as f:
            model = pickle.load(f)

    if hasattr(model, "model") and hasattr(model.model, "to"):
        model.model.to(map_location)

    if hasattr(model, "device"):
        model.device = torch.device(map_location)

    return model
