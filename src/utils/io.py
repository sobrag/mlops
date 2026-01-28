from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: Path) -> None:
    """
    Create a directory (and parents) if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_joblib(obj: Any, path: Path) -> None:
    """
    Save a Python object using joblib.
    """
    ensure_dir(path.parent)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """
    Load a Python object saved with joblib.
    """
    return joblib.load(path)


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary to a JSON file.
    """
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """
    Load a dictionary from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
