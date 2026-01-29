import importlib


def test_modules_importable():
    modules = [
        "src.models.train",
        "src.models.calibrate",
        "src.models.predict",
        "src.models.evaluate",
    ]

    for m in modules:
        importlib.import_module(m)
