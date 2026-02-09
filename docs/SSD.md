# System Specification Document (SSD)

**Project Title:** News Credibility Estimation  
**Version:** 1.0

**Team:**
- **Sofia Bragagnolo** – Project Manager  
- **Ester De Giosa** – Data Scientist  
- **Alina Fogar** – MLOps Engineer  
- **Riccardo Samaritan** – Software Developer  

- [System Specification Document (SSD)](#system-specification-document-ssd)
  - [Buisness Problem](#buisness-problem)
  - [Machine Learning Problem Formulation](#machine-learning-problem-formulation)
    - [Input Dataset](#input-dataset)
    - [Model](#model)
    - [Output](#output)
  - [Key Performance Indicators (KPIs)](#key-performance-indicators-kpis)
  - [Data Specification](#data-specification)
    - [Data Sources](#data-sources)
    - [Data Flow](#data-flow)
    - [Data Quality](#data-quality)
    - [Data Preprocessing](#data-preprocessing)
  - [Functional Requirments](#functional-requirments)
  - [Non-functional Requirments](#non-functional-requirments)
  - [Project Architecture](#project-architecture)
    - [Training](#training)
    - [Validation](#validation)
    - [Deployment](#deployment)
    - [Monitoring](#monitoring)
  - [Risk Analysis](#risk-analysis)

## Buisness Problem
The diffusion of fake news represents a major societal challenge. Online platforms, news aggregators and end users are increasingly exposed to unreliable or misleading information, making it difficult to assess the credibility of news.
From a business standpoint, this type of systems could support content moderation pipelines for social media platforms, browser extensions or APIs that provide credibility feedback to end users, internal quality checks (content quality scoring, ranking support) for news aggregators.
This project focuses on the technical feasibility and lifecycle management of such a system rather than on direct commercial deployment.

## Machine Learning Problem Formulation
### Input Dataset
The system uses the WELFake dataset, a publicly available dataset containing news articles labeled as real or fake. WELFake is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. 
Each instance includes:

| Title: headline of the article | Content: full article body | Label: binary label (0 = fake, 1 = real) |
|--------------------------------|-------------------------|------------------------------------------|

Example:

| Title                                                                 | Content                                                                                              | Label |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------|
| Lithuania gets minority government as junior partner leaves            | VILNIUS (Reuters) - Lithuania’s government lost its majority in parliament on Saturday after its jun...   | 0         |
| COLLEGE REPUBLICANS PRESIDENT Attacked by Antifa: “Like a pack of wolves” | So much for trying to come together like the left always says they think we should do. It was a brave... | 1         |
| For Helping Immigrants, Chobani’s Founder Draws Threats - The New York Times | By many measures, Chobani embodies the classic American immigrant success story. Its founder... | 0
...


### Model
The machine learning component is formulated as a supervised text classification problem.
The baseline model is a logistic Regression trained on TF-IDF representations of the text.

### Output 
The model outputs a probability associated with the predicted class (real vs fake). This probability is interpreted as a credibility score, rather than a strict binary decision:
- low score → low credibility (likely fake)
- high score → high credibility (likely real)

## Key Performance Indicators (KPIs)
The system is evaluated using both predictive performance and operational stability metrics.
Primary KPIs are Accuracy and F1-score (to handle potential class imbalance). 
Operational KPIs include stability of model predictions over time, distribution consistency of credibility scores and rift detection signals on input data and outputs.


## Data Specification
### Data Sources 
The WELFake dataset can be downloaded from Kaggle (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). 

### Data Flow
![Data Flow](images/dataflow.drawio.png)

### Data Quality 
The dataset presents typical issues found in real-world text data (missing or empty text fields, duplicated articles, noisy formatting and punctuation). These issues are explicitly handled during preprocessing and documented as system limitations.

### Data Preprocessing 
Preprocessing is implemented as a deterministic pipeline and includes:
- lowercasing
- whitespace normalization
- regex-based text cleaning
- removal of missing values and duplicates
- TF-IDF vectorization

The preprocessing logic is modular, reusable, and reproducible.

## Functional Requirments
The system must satisfy the following functional requirements:

- The system must use the WELFake dataset
- The system must preprocess text before training
- The system must split data into train/validation/test sets
- A trained credibility estimation model is available and produces reproducible results
- The model must output a credibility score
- The system must support model retraining
- The trained model and preprocessing artifacts can be reused consistently across training and inference

## Non-functional Requirments
- Data preprocessing must be reproducible
- Training and inference results must be reproducible given the same data and configuration
- The system must support batch processing
- Data quality issues must be documented
- Datasets must be versioned
- Model training must be reproducible
- The system architecture must allow easy model updates and monitoring extensions.
- The system must ensure data consistency across runs (monitoring of model behaviour)
- System behavior, input data, and model outputs must be observable through monitoring tools
- The model must achieve acceptable baseline performance
- Inference latency must be acceptable for interactive use in a web-based API

## Project Architecture

### Training
### Validation

### Deployment

### Monitoring

## Risk Analysis