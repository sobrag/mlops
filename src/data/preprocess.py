import pandas as pd
import re
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by converting to lowercase and removing extra whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove multiple spaces, tabs, newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for training: select columns, clean, create full_text.
    
    Args:
        df: Raw dataframe from load_welfake()
        
    Returns:
        pd.DataFrame: Prepared dataframe with columns: title, text, label, full_text, clean_text
    """
    # Select relevant columns
    required_columns = ['title', 'text', 'label']
    
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df[required_columns].copy()
    
    # Remove rows with missing values
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_rows - len(df)} rows with missing values")
    
    # Combine title and text into full_text
    df['full_text'] = df['title'].str.strip() + ". " + df['text'].str.strip()
    
    # Apply text cleaning
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    # Convert labels to int
    df['label'] = df['label'].astype(int)
    
    # Remove duplicates based on full_text
    initial_rows = len(df)
    df = df.drop_duplicates(subset='full_text')
    logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
    
    logger.info(f"Final dataset: {len(df)} rows")
    logger.info(f"Class distribution:\n{df['label'].value_counts()}")
    
    # aggiunto
    if df.empty:
        raise ValueError("Dataset is empty after preprocessing")

    return df


def get_dataset_metadata(df: pd.DataFrame) -> Dict:
    """
    Extract metadata about the dataset.
    
    Args:
        df: Prepared dataframe
        
    Returns:
        dict: Metadata including row counts, class distribution, etc.
    """
    metadata = {
        'total_rows': len(df),
        'columns': df.columns.tolist(),
        'class_distribution': df['label'].value_counts().to_dict(),
        'class_balance': df['label'].value_counts(normalize=True).to_dict(),
        'missing_values': df.isnull().sum().to_dict()
    }
    
    return metadata


def split_features_labels(df: pd.DataFrame, text_column: str = 'clean_text') -> Tuple[pd.Series, pd.Series]:
    """
    Extract features (text) and labels from dataframe.
    
    Args:
        df: Prepared dataframe
        text_column: Name of the column containing text data (default: 'clean_text')
        
    Returns:
        Tuple of (X: texts, y: labels)
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")
    
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in dataframe")
    
    X = df[text_column]
    y = df['label']
    
    logger.info(f"Extracted {len(X)} samples with {y.nunique()} unique labels")
    
    return X, y


def preprocess_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline: clean, prepare, extract metadata.
    
    Args:
        df_raw: Raw dataframe from load_welfake()
        
    Returns:
        Tuple of (prepared_df, metadata)
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Prepare dataset
    df_prepared = prepare_dataset(df_raw)
    
    # Get metadata
    metadata = get_dataset_metadata(df_prepared)
    
    logger.info("Preprocessing pipeline complete")
    logger.info(f"Metadata: {metadata}")
    
    return df_prepared, metadata


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from src.data.load_data import load_csv

    parser = argparse.ArgumentParser(description="Run preprocessing on a CSV file.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_welfake.csv",
        help="Path to input CSV (default: data/sample_welfake.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. "
            "Use --input to point to an existing CSV (e.g., data/sample_welfake.csv)."
        )

    df_raw = load_csv(input_path)
    df_prepared, metadata = preprocess_pipeline(df_raw)

    print("\n=== Preprocessing Results ===")
    print(f"Total rows: {metadata['total_rows']}")
    print(f"Class distribution: {metadata['class_distribution']}")
    print(f"\nFirst few rows:\n{df_prepared[['clean_text', 'label']].head()}")

    X, y = split_features_labels(df_prepared)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
