"""Lightweight continuous training trigger with drift detection.

This job loads the latest model and vectorizer artifacts, computes drift metrics on
new batches versus reference statistics, logs results to W&B, and can trigger a
retrain when thresholds are exceeded.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

from pipelines.inference_pipeline import _latest_run_dir, _load_artifacts
from src.config import Config
from src.data.load_data import load_csv
from src.data.preprocess import clean_text
from src.models.predict import predict_proba_real
from src.utils.io import ensure_dir, save_json


logger = logging.getLogger(__name__)


# Config helpers


def _load_raw_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _drift_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "top_k_tokens": 200,
        "psi_buckets": 10,
        "psi_threshold": 0.25,
        "js_threshold": 0.05,
        "max_rows_batch": 5000,
        "auto_retrain": False,
        "retrain_config": "configs/train.yml",
    }
    cfg = raw.get("drift", {}) or {}
    return {**defaults, **cfg}


def _path_config(raw: Dict[str, Any]) -> Dict[str, Path]:
    paths = raw.get("paths", {}) or {}
    return {
        "artifacts_dir": Path(paths.get("artifacts_dir", "artifacts")),
        "incoming_glob": Path(paths.get("incoming_glob", "data/incoming/*.csv")),
        "reference_stats": Path(paths.get("reference_stats", "artifacts/reference_stats.json")),
        "results_dir": Path(paths.get("results_dir", "results")),
        "raw_data_path": Path(paths.get("raw_data_path", "data/WELFake_dataset.csv")),
    }


def _logging_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "use_wandb": True,
        "wandb_project": "mlops",
        "wandb_entity": None,
        "wandb_mode": os.getenv("WANDB_MODE", "online"),
    }
    cfg = raw.get("logging", {}) or {}
    return {**defaults, **cfg}


def _split_cfg(raw: Dict[str, Any]) -> Dict[str, float]:
    split = raw.get("split", {}) or {}
    return {
        "reference_frac": float(split.get("reference_frac", 0.7)),
        "drift_start_frac": float(split.get("drift_start_frac", 0.8)),
        "drift_end_frac": float(split.get("drift_end_frac", 1.0)),
    }


# Stats & metrics


def _text_lengths(texts: pd.Series) -> np.ndarray:
    return texts.astype(str).str.split().map(len).to_numpy()


def _token_freq(texts: pd.Series, analyzer, top_k: int) -> Dict[str, float]:
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(analyzer(str(t)))
    most_common = counts.most_common(top_k)
    total = sum(c for _, c in most_common) or 1
    return {tok: c / total for tok, c in most_common}


def _hist_proportions(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum() or 1
    return counts / total


def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    eps = 1e-12
    expected = np.where(expected == 0, eps, expected)
    actual = np.where(actual == 0, eps, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    tokens = set(p.keys()) | set(q.keys())
    if not tokens:
        return 0.0
    eps = 1e-12
    p_arr = np.array([p.get(tok, 0.0) for tok in tokens], dtype=float) + eps
    q_arr = np.array([q.get(tok, 0.0) for tok in tokens], dtype=float) + eps
    p_arr /= p_arr.sum()
    q_arr /= q_arr.sum()
    m = 0.5 * (p_arr + q_arr)
    return 0.5 * float(np.sum(p_arr * np.log(p_arr / m)) + np.sum(q_arr * np.log(q_arr / m)))


def _prepare_texts(df_raw: pd.DataFrame, cfg: Config) -> pd.Series:
    missing = [c for c in [cfg.title_col, cfg.text_col] if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns for drift check: {missing}")
    df = df_raw[[cfg.title_col, cfg.text_col]].copy()
    df[cfg.title_col] = df[cfg.title_col].fillna("").astype(str).str.strip()
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str).str.strip()
    combined = df[cfg.title_col] + ". " + df[cfg.text_col]
    return combined.apply(clean_text)


def _build_bins(values: np.ndarray, buckets: int) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, buckets + 1)


def compute_reference_stats(texts: pd.Series, vectorizer, model, drift_cfg: Dict[str, Any]) -> Dict[str, Any]:
    lengths = _text_lengths(texts)
    len_bins = _build_bins(lengths, drift_cfg["psi_buckets"])
    len_dist = _hist_proportions(lengths, len_bins)

    X = vectorizer.transform(texts.to_numpy())
    probs = predict_proba_real(model, X)
    prob_bins = np.linspace(0.0, 1.0, drift_cfg["psi_buckets"] + 1)
    prob_dist = _hist_proportions(probs, prob_bins)

    analyzer = vectorizer.vectorizer.build_analyzer()
    token_freq = _token_freq(texts, analyzer, drift_cfg["top_k_tokens"])

    return {
        "len_bins": len_bins.tolist(),
        "len_dist": len_dist.tolist(),
        "prob_bins": prob_bins.tolist(),
        "prob_dist": prob_dist.tolist(),
        "token_freq": token_freq,
        "rows": int(len(texts)),
    }


def compute_batch_stats(texts: pd.Series, vectorizer, model, ref_stats: Dict[str, Any], drift_cfg: Dict[str, Any]) -> Dict[str, Any]:
    lengths = _text_lengths(texts)
    len_bins = np.array(ref_stats["len_bins"])
    len_dist = _hist_proportions(lengths, len_bins)

    X = vectorizer.transform(texts.to_numpy())
    probs = predict_proba_real(model, X)
    prob_bins = np.array(ref_stats["prob_bins"])
    prob_dist = _hist_proportions(probs, prob_bins)

    analyzer = vectorizer.vectorizer.build_analyzer()
    token_freq = _token_freq(texts, analyzer, drift_cfg["top_k_tokens"])

    return {
        "len_dist": len_dist.tolist(),
        "prob_dist": prob_dist.tolist(),
        "token_freq": token_freq,
        "rows": int(len(texts)),
    }


def compute_drift(ref_stats: Dict[str, Any], new_stats: Dict[str, Any], drift_cfg: Dict[str, Any]) -> Dict[str, Any]:
    psi_len = population_stability_index(np.array(ref_stats["len_dist"]), np.array(new_stats["len_dist"]))
    psi_pred = population_stability_index(np.array(ref_stats["prob_dist"]), np.array(new_stats["prob_dist"]))
    js_tok = js_divergence(ref_stats["token_freq"], new_stats["token_freq"])

    alert = (psi_len > drift_cfg["psi_threshold"]) or (psi_pred > drift_cfg["psi_threshold"]) or (js_tok > drift_cfg["js_threshold"])
    score = max(psi_len, psi_pred, js_tok)

    return {
        "psi_length": psi_len,
        "psi_pred": psi_pred,
        "js_tokens": js_tok,
        "score": score,
        "alert": alert,
    }


# I/O helpers


def _latest_file(glob_pattern: Path) -> Path:
    matches = sorted(Path().glob(str(glob_pattern)), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files match glob: {glob_pattern}")
    return matches[0]


def _split_reference_and_tail(df: pd.DataFrame, split_cfg: Dict[str, float]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    ref_end = int(n * split_cfg["reference_frac"])
    tail_start = int(n * split_cfg["drift_start_frac"])
    tail_end = int(n * split_cfg["drift_end_frac"])
    ref = df.iloc[:ref_end]
    tail = df.iloc[tail_start:tail_end]
    return ref, tail


def _load_reference_texts(df_all: pd.DataFrame, cfg: Config, split_cfg: Dict[str, float]) -> pd.Series:
    ref_df, _ = _split_reference_and_tail(df_all, split_cfg)
    if len(ref_df) == 0:
        raise ValueError("Reference dataset is empty")
    return _prepare_texts(ref_df, cfg)


def _load_new_batch_texts(batch_path: Path, cfg: Config, max_rows: int | None) -> pd.Series:
    df = load_csv(batch_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()
    return _prepare_texts(df, cfg)


def _save_report(report: Dict[str, Any], output_dir: Path) -> Path:
    ensure_dir(output_dir)
    path = output_dir / "drift_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def _resolve_dataset_file(cfg: Config, path_cfg: Dict[str, Path], log_cfg: Dict[str, Any]) -> Path:
    """Resolve dataset path, preferring local file, otherwise W&B artifact if configured."""
    local_path = Path(cfg.raw_data_path)
    if local_path.exists():
        return local_path

    dataset_artifact = getattr(cfg, "dataset_artifact", None)
    use_wandb = bool(log_cfg.get("use_wandb", False))
    wandb_mode = log_cfg.get("wandb_mode", os.getenv("WANDB_MODE", "online"))

    if dataset_artifact and use_wandb and wandb_mode != "offline":
        import wandb

        run = wandb.init(
            project=log_cfg.get("wandb_project"),
            entity=log_cfg.get("wandb_entity"),
            job_type="ct-dataset",
            mode=wandb_mode,
        )
        art = run.use_artifact(dataset_artifact)
        download_dir = Path(art.download())
        csvs = list(download_dir.glob("*.csv"))
        run.finish()
        if not csvs:
            raise FileNotFoundError(f"No CSV found in downloaded artifact {dataset_artifact}")
        return csvs[0]

    raise FileNotFoundError(
        f"Dataset not found locally at {local_path} and unable to resolve artifact "
        f"(dataset_artifact={dataset_artifact}, use_wandb={use_wandb}, wandb_mode={wandb_mode})."
    )


# W&B logging


def _log_wandb(enabled: bool, log_cfg: Dict[str, Any], report: Dict[str, Any], report_path: Path) -> None:
    if not enabled:
        return
    import wandb

    run_name = log_cfg.get("run_name") or f"ct_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    tags = ["drift"]
    split_cfg = report.get("config", {}).get("split") or {}
    if all(k in split_cfg for k in ("reference_frac", "drift_start_frac", "drift_end_frac")):
        tags.append(
            f"split_{split_cfg['reference_frac']}_{split_cfg['drift_start_frac']}_{split_cfg['drift_end_frac']}"
        )
    dataset_tag = None
    dataset_path = report.get("dataset_path")
    if dataset_path:
        dataset_tag = f"dataset:{Path(dataset_path).stem}"
    if dataset_tag:
        tags.append(dataset_tag)

    run = wandb.init(
        project=log_cfg.get("wandb_project"),
        entity=log_cfg.get("wandb_entity"),
        job_type="ct",
        mode=log_cfg.get("wandb_mode", os.getenv("WANDB_MODE", "online")),
        name=run_name,
        tags=tags,
    )

    wandb.log(
        {
            "drift/psi_length": report["metrics"]["psi_length"],
            "drift/psi_pred": report["metrics"]["psi_pred"],
            "drift/js_tokens": report["metrics"]["js_tokens"],
            "drift/score": report["metrics"]["score"],
            "drift/alert": report["metrics"]["alert"],
            "rows/reference": report["reference"]["rows"],
            "rows/new": report["new"]["rows"],
        }
    )

    art = wandb.Artifact(name="drift_report", type="drift")
    art.add_file(str(report_path))
    wandb.log_artifact(art)

    if report["metrics"]["alert"]:
        wandb.alert(
            title="Data drift detected",
            text=f"score={report['metrics']['score']:.3f}, psi_len={report['metrics']['psi_length']:.3f}, psi_pred={report['metrics']['psi_pred']:.3f}, js={report['metrics']['js_tokens']:.3f}",
        )

    run.finish()


# Retrain hook


def _trigger_retrain(enabled: bool, retrain_cfg_path: str) -> None:
    if not enabled:
        return
    cmd = ["python", "pipelines/train_pipeline.py", "--config", retrain_cfg_path]
    logger.info(f"Starting retrain: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# Entrypoint


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    raw_cfg = _load_raw_yaml(Path(args.config))
    drift_cfg = _drift_config(raw_cfg)
    path_cfg = _path_config(raw_cfg)
    log_cfg = _logging_config(raw_cfg)
    split_cfg = _split_cfg(raw_cfg)

    cfg = Config.from_yaml(args.config)

    if args.incoming:
        batch_path = Path(args.incoming)
    else:
        batch_path = _latest_file(path_cfg["incoming_glob"])

    logger.info(
        f"Drift start | config={args.config} | batch={batch_path} | artifacts_dir={path_cfg['artifacts_dir']} "
        f"| wandb={log_cfg.get('use_wandb')} ({log_cfg.get('wandb_mode')})"
    )

    dataset_path = _resolve_dataset_file(cfg, path_cfg, log_cfg)
    df_all = load_csv(dataset_path)

    # Load reference stats or build them
    run_dir = _latest_run_dir(path_cfg["artifacts_dir"])
    vectorizer, model = _load_artifacts(run_dir)

    ref_stats_path = path_cfg["reference_stats"]
    if ref_stats_path.exists():
        ref_stats = json.loads(ref_stats_path.read_text())
        logger.info(f"Reference stats: loaded ({ref_stats_path})")
    else:
        ref_texts = _load_reference_texts(df_all, cfg, split_cfg)
        ref_stats = compute_reference_stats(ref_texts, vectorizer, model, drift_cfg)
        ensure_dir(ref_stats_path.parent)
        save_json(ref_stats, ref_stats_path)
        logger.info(f"Reference stats: created ({ref_stats_path})")

    new_texts = _load_new_batch_texts(batch_path, cfg, drift_cfg.get("max_rows_batch"))
    new_stats = compute_batch_stats(new_texts, vectorizer, model, ref_stats, drift_cfg)
    metrics = compute_drift(ref_stats, new_stats, drift_cfg)

    output_dir = path_cfg["results_dir"] / "drift" / datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "reference": ref_stats,
        "new": new_stats,
        "metrics": metrics,
        "run_dir": str(run_dir),
        "batch_path": str(batch_path),
        "dataset_path": str(dataset_path),
        "config": {
            "config_path": args.config,
            "drift": drift_cfg,
            "split": split_cfg,
            "paths": {k: str(v) for k, v in path_cfg.items()},
        },
    }

    report_path = _save_report(report, output_dir)
    logger.info(
        f"Drift | alert={metrics['alert']} score={metrics['score']:.4f} "
        f"(psi_len={metrics['psi_length']:.4f}, psi_pred={metrics['psi_pred']:.4f}, js={metrics['js_tokens']:.4f}) "
        f"| batch={batch_path} | reference_rows={ref_stats['rows']} | new_rows={new_stats['rows']} | report={report_path}"
    )

    _log_wandb(log_cfg.get("use_wandb", False), log_cfg, report, report_path)

    if metrics["alert"] and drift_cfg.get("auto_retrain", False):
        _trigger_retrain(True, drift_cfg.get("retrain_config", "configs/train.yml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drift detection + CT trigger")
    parser.add_argument("--config", type=str, default="configs/ct.yml", help="Path to CT/drift config YAML")
    parser.add_argument("--incoming", type=str, default=None, help="Optional path to a specific incoming CSV")
    args = parser.parse_args()
    main(args)
