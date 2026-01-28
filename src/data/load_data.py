import pandas as pd
from pathlib import Path

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# aggiungere if file non sono qua ecc