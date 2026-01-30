import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_csv(path: Path | str) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame with error handling.
    
    Args:
        path: Path to CSV file (Path object or string)
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid CSV
    """
    # Convert to Path if string
    path = Path(path)
    
    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please ensure the file exists or check your config."
        )
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    # Check file extension
    if path.suffix.lower() != '.csv':
        logger.warning(f"File extension is '{path.suffix}', expected '.csv'")
    
    try:
        # Load CSV
        logger.info(f"Loading dataset from: {path}")
        df = pd.read_csv(path)
        
        # Check if dataframe is empty
        if df.empty:
            raise ValueError(f"CSV file is empty: {path}")
        
        # Log success
        logger.info(f"Successfully loaded {len(df)} rows from {path.name}")
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty or invalid: {path}")
    
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse CSV file {path}: {e}")
    
    except Exception as e:
        raise RuntimeError(f"Unexpected error loading {path}: {e}")