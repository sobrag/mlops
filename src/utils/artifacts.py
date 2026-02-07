from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.features.vectorize import TextVectorizer
from src.utils.io import load_joblib


def _find_local_model_dir(artifacts_dir: Path) -> Optional[Path]:
    """Return the latest dir in artifacts_dir containing model and vectorizer."""
    if not artifacts_dir.exists():
        return None
    candidates = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    for d in sorted(candidates, reverse=True):
        if (d / "model.joblib").exists() and (d / "vectorizer.joblib").exists():
            return d
    return None


def _download_model_artifact(
    model_artifact: str,
    artifacts_dir: Path,
    use_wandb: bool,
    wandb_mode: str,
    project: Optional[str],
    entity: Optional[str],
) -> Path:
    if not model_artifact:
        raise FileNotFoundError("No model_artifact configured")
    if not use_wandb:
        raise FileNotFoundError("W&B disabled, cannot download model artifact")
    if wandb_mode == "offline":
        raise FileNotFoundError("WANDB_MODE=offline, cannot download model artifact")

    import wandb

    dest = artifacts_dir / "_downloaded" / model_artifact.replace(":", "_")
    dest.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="model-download",
        mode=wandb_mode or "online",
        name=f"download_{model_artifact}",
    )
    art = run.use_artifact(model_artifact)
    art.download(root=str(dest))
    run.finish()
    return dest


def resolve_model_dir(
    artifacts_dir: Path,
    model_artifact: Optional[str] = None,
    use_wandb: bool = False,
    wandb_mode: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
) -> Path:
    """Return a directory containing model.joblib and vectorizer.joblib.

    Prefer local artifacts_dir; fallback to downloading a W&B model artifact.
    """
    local = _find_local_model_dir(artifacts_dir)
    if local:
        return local
    return _download_model_artifact(model_artifact, artifacts_dir, use_wandb, wandb_mode or "online", project, entity)


def load_model_bundle(model_dir: Path):
    vec_path = model_dir / "vectorizer.joblib"
    model_path = model_dir / "model.joblib"
    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Model bundle incomplete in {model_dir}")
    vectorizer = TextVectorizer.load(vec_path)
    model = load_joblib(model_path)
    return vectorizer, model
