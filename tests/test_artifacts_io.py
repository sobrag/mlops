from pathlib import Path
from src.utils.io import save_json, load_json, save_joblib, load_joblib

def test_io_roundtrip(tmp_path: Path):
    d = {"a": 1}
    p = tmp_path / "x.json"
    save_json(d, p)
    assert load_json(p) == d

    obj = {"k": 2}
    m = tmp_path / "m.joblib"
    save_joblib(obj, m)
    assert load_joblib(m) == obj
