import wandb
from pathlib import Path

def main():
    # Login to W&B
    wandb.login()
    
    # Initialize run
    run = wandb.init(
        project="mlops",
        job_type="upload-dataset",
        name="upload-welfake-dataset"
    )
    
    # Create artifact
    artifact = wandb.Artifact(
        name="welfake_dataset",
        type="dataset",
        description="WELFake fake news detection dataset",
        metadata={
            "source": "https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification",
            "rows": "72134",  
            "columns": ["title", "text", "label"]
        }
    )
    
    # Add dataset file
    dataset_path = Path("data/WELFake_dataset.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please download it first."
        )
    
    artifact.add_file(str(dataset_path))
    
    # Log artifact
    run.log_artifact(artifact)
    run.finish()
    
    print("Dataset uploaded successfully!")
    print("   Artifact: welfake_dataset:latest")

if __name__ == "__main__":
    main()