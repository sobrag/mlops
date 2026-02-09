# System Specification Document (SSD)

## News Credibility Estimation

**System Version:** 2.0

**Last updated:** 2026‑02‑09

**Team:**
- **Sofia Bragagnolo** – Project Manager  
- **Ester De Giosa** – Data Scientist  
- **Alina Fogar** – MLOps Engineer  
- **Riccardo Samaritan** – Software Developer  

---

## TABLE OF CONTENTS
- [System Specification Document (SSD)](#system-specification-document-ssd)
  - [News Credibility Estimation](#news-credibility-estimation)
  - [TABLE OF CONTENTS](#table-of-contents)
  - [Business Problem](#business-problem)
  - [Definitions, Acronyms, and Abbreviations](#definitions-acronyms-and-abbreviations)
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
  - [Functional Requirements](#functional-requirements)
  - [FR-1: Data Management](#fr-1-data-management)
    - [FR-2: Model Training](#fr-2-model-training)
  - [Non-functional Requirements](#non-functional-requirements)
  - [Project Architecture](#project-architecture)
    - [Training](#training)
    - [Validation](#validation)
    - [Deployment](#deployment)
    - [Monitoring](#monitoring)
  - [Risk Analysis](#risk-analysis)

## Business Problem

The diffusion of fake news represents a major societal challenge. Online platforms, news aggregators, and end users are increasingly exposed to unreliable or misleading information, making it difficult to assess the credibility of news.

From a business standpoint, such a system could support content moderation pipelines for social media platforms, browser extensions, or APIs that provide credibility feedback to end users, as well as internal quality checks (e.g., content quality scoring and ranking support) for news aggregators. Rather than attempting binary classification (which can be too rigid and prone to false positives), the system provides a nuanced credibility score that helps users make informed decisions about the reliability of content they encounter.

This project focuses on the technical feasibility and lifecycle management of such a system rather than on direct commercial deployment.

## Definitions, Acronyms, and Abbreviations

**Definitions**
- **Credibility score**: A score derived from the model probability `P(real | x)` scaled to 0–100.
- **Model bundle**: A versioned set of artifacts produced by training, including the trained model, fitted vectorizer, and run metadata (e.g., metrics and configuration).
- **Drift report**: A JSON report produced by the drift detection job, summarizing input/output distribution shifts for a new batch compared to reference statistics.
- **Reference statistics**: Baseline statistics computed from a reference slice of data and reused to compare incoming batches over time.
- **Batch inference**: Running predictions over multiple records (e.g., a CSV file) in one execution or one API request.

**Acronyms and abbreviations**
- **SSD**: System Specification Document.
- **WELFake**: Public dataset used for training and evaluation, with labels `0=real`, `1=fake`.
- **TF-IDF**: Term Frequency–Inverse Document Frequency, the feature representation used for text.
- **LR**: Logistic Regression, the selected baseline classifier.
- **API**: Application Programming Interface (Flask service exposing `/predict` and `/predict/batch`).
- **CI/CD**: Continuous Integration / Continuous Delivery (GitHub Actions workflows validating code and containers).
- **W&B**: Weights & Biases, used for optional artifact and experiment tracking.
- **PSI**: Population Stability Index, used to measure distribution shift (e.g., text length and predicted probability distributions).
- **JS divergence**: Jensen–Shannon divergence, used to quantify token-frequency distribution changes.
- **Prometheus**: Metrics collection system scraping the API `/metrics` endpoint.
- **Grafana**: Dashboarding system visualizing Prometheus metrics.

## Machine Learning Problem Formulation

### Input Dataset

The system uses the **WELFake dataset**, a publicly available dataset containing news articles labeled as _real_ or _fake_. 

WELFake contains 72,134 news articles with 35,028 real and 37,106 fake news. 
Each instance includes:

| Title: headline of the article | Content: full article body | Label: binary label (0 = real, 1 = fake) |
|--------------------------------|-------------------------|------------------------------------------|

We identified an inaccuracy in the Kaggle dataset documentation, where the meanings of labels 0 and 1 were incorrectly interchanged.

Example rows from the dataset:

| Title                                                                 | Content                                                                                              | Label |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------|
| Lithuania gets minority government as junior partner leaves            | VILNIUS (Reuters) - Lithuania’s government lost its majority in parliament on Saturday after its jun...   | 0         |
| COLLEGE REPUBLICANS PRESIDENT Attacked by Antifa: “Like a pack of wolves” | So much for trying to come together like the left always says they think we should do. It was a brave... | 1         |
| For Helping Immigrants, Chobani’s Founder Draws Threats - The New York Times | By many measures, Chobani embodies the classic American immigrant success story. Its founder... | 0


### Model

The machine learning component is formulated as a supervised text classification problem.
The selected model is a **Logistic Regression** classifier trained on **TF-IDF representations** of the input text.

TF-IDF is a statistical technique used to convert textual data into numerical feature vectors suitable for machine learning models.
It measures how important a word is to a document relative to the entire text and is composed of two factors:
- **Term Frequency (TF)**: measures how frequently a term appears in a document, capturing its relevance within that specific article.
- **Inverse Document Frequency (IDF)**: down-weights terms that appear frequently across many documents and assigns higher importance to more important terms.

### Output 

For each news article, the model outputs the probability of the _Real News_ class, `P(real | x)`. We use this probability as a continuous credibility score (rather than a hard real/fake decision), and map it to a 0–100 score via `credibility_score = P(real | x) * 100`.


A higher score reflects a greater probability that the news article is genuine and thus credible, whereas a lower score suggests a higher likelihood of the article being fake.

In addition to the credibility score, the system also provides a binary classification output (real vs fake) based on a configurable decision threshold.

## Key Performance Indicators (KPIs)

The system is evaluated using predictive performance metrics, operational stability indicators, and monitoring KPIs to ensure reliable and trustworthy operation throughout its lifecycle.

**ML Model Performance KPIs**:

The following metrics provide a comprehensive evaluation of classification performance:
- **Accuracy**: Overall correct classification rate across real and fake news.
- **F1-Score**: Harmonic mean of precision and recall, particularly critical given potential class imbalance.

**Operational Performance KPIs**:

System-level metrics monitor real-time performance and reliability:
- **API Response Time**: a low latency for single predictions ensures responsive user experience.
- **System Availability**: Uptime percentage excluding planned maintenance.
- **Throughput**: Request handling capacity under load.
- **API Error Rate**: Percentage of failed predictions requiring attention.

**Data and Model Drift KPIs**:

Monitoring metrics detect distribution shifts and model degradation over time:
- **Population Stability Index (PSI)**: Measures distribution shift in text length and prediction scores.
- **Jensen-Shannon Divergence**: Detects changes in token frequency distributions.
- **Prediction Distribution Shift**: Monitors changes in average credibility scores.
- **Text Length Drift**: Tracks changes in input article characteristics.

These KPIs form a comprehensive evaluation framework spanning model accuracy, operational efficiency, and long-term system stability. Regular monitoring ensures the system maintains both predictive reliability and operational trustworthiness throughout production deployment.

## Data Specification

### Data Sources 

The WELFake dataset can be downloaded from Kaggle (https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). 

### Data Flow
![Data Flow](images/dataflow.drawio.png)

### Data Quality 

The dataset presents typical issues found in real-world text data, including missing or empty text fields, duplicated articles, noisy formatting and punctuation.

These issues are systematically addressed during the preprocessing phase through data cleaning and filtering steps. Remaining limitations related to data quality are acknowledged and considered during model evaluation and monitoring.

### Data Preprocessing 

Data preprocessing is implemented as a deterministic and reproducible pipeline applied consistently across training, validation, and inference phases.

The preprocessing steps include:

- lowercasing of text
- whitespace normalization
- regex-based text cleaning
- removal of missing values and duplicates entries
- TF-IDF vectorization of the cleaned text

The preprocessing logic is modular, reusable, and fully reproducible, ensuring consistency between model training and deployment.

## Functional Requirements

The system must satisfy the following functional requirements:

## FR-1: Data Management

| ID | Requirement |
|---|---|
| **FR-1.1** | The system shall load the WELFake dataset from local storage or Weights & Biases artifacts. |
| **FR-1.2** | The system shall apply a deterministic text preprocessing pipeline (lowercase, regex cleaning, whitespace normalization) before model training and inference. |
| **FR-1.3** | The system shall remove articles with missing text or duplicate content |
| **FR-1.4** | The system shall split data into train, validation, and test sets with stratified sampling |
| **FR-1.4** | The system shall train a machine learning model for news credibility estimation and produce reproducible results given the same data and configuration. |
| **FR-1.5** | The system shall support dataset versioning through Weights & Biases artifacts when enabled |

### FR-2: Model Training



| **FR-1.5** | The model shall output a credibility score expressed as a probability associated with the _Real News_ class. |
| **FR06** | The system shall support model retraining to incorporate new data or updated configurations. |
| **FR07** | The trained model and preprocessing artifacts shall be reusable consistently applied across training and inference pipelines. |

## Non-functional Requirements

| ID | Requirement |
|---|---|
| **NFR01** | Data preprocessing, model training, and inference shall be reproducible given the same input data, configuration parameters, and random seed.
| **NFR02** | Datasets, preprocessing artifacts, and trained models shall be versioned to ensure traceability and reproducibility across experiments and deployments.
| **NFR03** | The system shall support batch processing for both training and inference workflows.
| **NFR04** | Data quality issues and known limitations shall be explicitly documented and handled within the preprocessing pipeline.
| **NFR05** | The system architecture shall support modular model updates and allow future extensions for monitoring and retraining.
| **NFR06** | System behavior, input data distributions, and model outputs shall be observable through logging and monitoring mechanisms.
| **NFR07** | The system shall ensure consistency of model behavior across runs, enabling the detection of anomalous changes or degradation over time.
| **NFR08** | The trained model shall achieve an acceptable baseline predictive performance, suitable for demonstrating the feasibility of the proposed approach.
| **NFR09** | Inference latency shall be acceptable for interactive use, such as integration into a web-based API or user-facing application.

## Project Architecture

### Training
Training is implemented as a reproducible, configuration-driven pipeline that builds a complete “model bundle” for inference. The main entrypoint is `pipelines/train_pipeline.py` (typically executed as `python -m pipelines.train_pipeline --config configs/train.yml`) and it is designed to be fully repeatable given the same data and configuration.

At runtime, the pipeline resolves the dataset either from a local CSV file (`paths.raw_data_path`) and, when Weights & Biases (W&B) logging is enabled, by downloading a dataset artifact (`data.dataset_artifact`).
Raw data is loaded from CSV using `src/data/load_data.py` and processed through a deterministic preprocessing pipeline implemented in `src/data/preprocess.py`. This pipeline combines title and body text, normalizes and cleans textual content, removes missing entries, and eliminates duplicates articles.

Feature extraction is performed using TF-IDF vectorization (`src/features/vectorize.py`). A supervised classifier is then trained on the feature vectors (the default configuration uses Logistic Regression implemented in `src/models/train.py`). To improve the interpretability and reliability of the output probabilities as credibility scores, probability calibration is applied (`src/models/calibrate.py`). The calibrated model is subsequently evaluated on a validation split, and evaluation metrics are computed and persisted using `src/models/evaluate.py`.

Each training run produces a versioned output directory under `artifacts/run_<timestamp>/`, which includes the trained model, the fitted vectorizer, evaluation metrics, the effective configuration used for the run, and preprocessing metadata. When W&B logging is enabled, the same model bundle is also logged as a W&B artifact named `credibility_model_bundle`, with the aliases `latest` and `run/<run_id>` to support traceability and reproducibility.

### Validation
Validation happens in two places. First, during training, the pipeline computes model-level metrics on a held-out validation split and stores them in `metrics.json` inside each run directory (for example `artifacts/run_<timestamp>/metrics.json`). This provides a reproducible record of the model’s predictive performance for each versioned bundle.

Second, system-level validation is automated in CI. The GitHub Actions workflow on `main` (`.github/workflows/ci-full.yml`) runs linting and test suites, executes a smoke training run (to ensure the end-to-end pipeline still produces artifacts), runs a smoke drift check (to ensure drift reporting still executes), and validates the containerization layer by building the Docker images and bringing up `docker compose` services with health checks.

### Deployment
Deployment is implemented as a local Docker Compose stack (`docker-compose.yml`) that runs the full application and monitoring setup together. The stack includes: a Flask API served via Gunicorn (image built from `src/app/Dockerfile`), a Streamlit UI (image built from `src/ui/Dockerfile`), Prometheus (configured in `src/monitoring/prometheus/prometheus.yml`), and Grafana with provisioned datasources/dashboards (`src/monitoring/grafana/provisioning/`).

From an interaction perspective, the Streamlit UI calls the API for predictions, and Prometheus scrapes the API metrics endpoint. In the compose configuration the user-facing ports are exposed as: Streamlit on `8501`, API on `5001`, Prometheus on `9090`, and Grafana on `3000`. The API exposes `/health`, `/predict`, and `/predict/batch` for inference, and `/metrics` for Prometheus-compatible monitoring (via `prometheus-flask-exporter`).

At runtime, the API resolves the model bundle either from the local artifacts directory mounted into the container (`./artifacts`) or, when W&B integration is enabled, from a W&B artifact specified through environment variables (for example `MODEL_ARTIFACT=credibility_model_bundle:production` with `USE_WANDB=true`). In practice, deployment is a manual operation after a green CI run by starting/restarting the stack with `docker compose up -d --build` and selecting the desired model artifact alias through environment configuration.

### Monitoring
Monitoring is implemented in two complementary ways. For service observability, the API exports Prometheus-compatible metrics at `/metrics` using `prometheus-flask-exporter` (`src/app/main.py`). Prometheus scrapes these metrics as the `news_credibility_api` job targeting `api:5000` (`src/monitoring/prometheus/prometheus.yml`), and Grafana is provisioned to read from Prometheus and load dashboards automatically (`src/monitoring/grafana/provisioning/`).

For ML-specific monitoring, a batch drift detection job is implemented in `pipelines/ct_trigger.py` and configured via `configs/ct.yml`. The job compares incoming batches (for example CSVs under `data/incoming/`) against reference statistics derived from a reference slice of the dataset, and computes drift signals on both the inputs and the model output distribution. Concretely, it computes PSI on text length distribution, PSI on predicted probability distribution, and Jensen-Shannon divergence on token-frequency distributions. The outputs include reference statistics persisted to `artifacts/reference_stats.json` and a timestamped drift report JSON under `results/`. An optional auto-retraining path exists in the code but is disabled by default and gated by `auto_retrain` in `configs/ct.yml`.

## Risk Analysis
The system is designed for local deployment and reproducible experimentation; the table below summarizes the main risks and the mitigations that are already implemented.

| Risk | Impact | Implemented mitigation |
|---|---|---|
| Data quality / provenance (local CSV vs W&B artifacts; noisy/duplicated text) | Biased training and non-reproducible runs if inputs change | Deterministic preprocessing (`src/data/preprocess.py`); CI uses committed sample dataset (`data/sample_welfake.csv`) |
| Drift (input text and output probability distribution shifts) | Gradual performance degradation and unstable credibility scores | Drift checks in `pipelines/ct_trigger.py` with reports under `results/` and reference stats at `artifacts/reference_stats.json` |
| Manual deployment regressions (wrong config/model alias) | Shipping an unintended model or misconfigured stack | CI builds images and does `docker compose` bring-up with health checks (`.github/workflows/ci-full.yml`) |
| Limited production security / alerting (no auth, no automated alerts) | Unsafe if exposed publicly; delayed incident response | Intended local/offline usage; secrets via env vars must be managed by operators |
| Keyword stuffing / “trigger words” in the input text | Manipulated or unstable credibility scores (false positives/negatives) | Not implemented. Future work: add an input robustness layer (e.g., detect excessive token repetition / abnormal TF-IDF patterns and down-weight or flag the request). |