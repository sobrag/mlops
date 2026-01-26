### System Definition Document

## DATASET

The project uses the **WELFake dataset**, a publicly available collection of news articles labeled as real or fake. Download from Kaggle.
The dataset includes:
- Article title
- Full text
- Binary credibility label

This dataset is suitable for training an initial credibility estimation model, for simulating **production data streams** and studying changes in data distribution over time.


## Functional Requirments

- The system must use the WELFake dataset
- The system must split data into train/validation/test sets
- The system must preprocess text before training
- The model must output a credibility score
- The system must support model retraining

## Non-functional Requirments
- Data preprocessing must be reproducible
- Data quality issues must be documented
- Datasets must be versioned
- Datasets must be versioned
- Model training must be reproducible
- The system must ensure data consistency across runs (monitoring of model behaviour)
- The model must achieve acceptable baseline performance