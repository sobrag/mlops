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
- [Non-functional Requirements](#non-functional-requirements)
- [Project Architecture](#project-architecture)
  - [Training](#training)
  - [Validation](#validation)
  - [Deployment](#deployment)
  - [Monitoring](#monitoring)
- [Technology Selection Justification](#technology-selection-justification)
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

### FR-1: Data Management

| ID | Requirement |
|---|---|
| **FR-1.1** | The system shall load the WELFake dataset from local storage or Weights & Biases artifacts. |
| **FR-1.2** | The system shall apply a deterministic text preprocessing pipeline (lowercase, regex cleaning, whitespace normalization) before model training and inference. |
| **FR-1.3** | The system shall remove articles with missing text or duplicate content. |
| **FR-1.4** | The system shall split data into train, validation, and test sets with stratified sampling. |
| **FR-1.5** | The system shall support dataset versioning through Weights & Biases artifacts when enabled. |

### FR-2: Model Training

| ID | Requirement |
|---|---|
| **FR-2.1** | The system shall train a machine learning model for news credibility estimation and produce reproducible results given the same data and configuration. |
| **FR-2.2** | The system shall use a fixed random seed for reproducible results. |
| **FR-2.3** | The system shall serialize trained models, vectorizers, and configurations to disk and log them as Weights & Biases artifacts. |
| **FR-2.4** | The system shall compute and log evaluation metrics. |
| **FR-2.5** | The model shall output a credibility score expressed as a probability associated with the _Real News_ class. |
| **FR-2.6** | The system shall generate reference statistics for drift detection during training. |

### FR-3: Model Inference

| ID | Requirement |
|----|-------------|
| **FR-3.1** | The system shall expose a REST API for model predictions. | 
| **FR-3.2** | The system shall accept single article predictions via POST /predict endpoint. | 
| **FR-3.3** | The system shall accept batch predictions via POST /predict/batch endpoint. | 
| **FR-3.4** | The system shall return credibility scores (0-100) derived from model probabilities. | 
| **FR-3.5** | The system shall load model artifacts from local storage or Weights & Biases. | 
| **FR-3.6** | The system shall validate input text is non-empty before prediction. |

### FR-4: User Interface

| ID | Requirement |
|----|-------------|
| **FR-4.1** | The system shall provide a Streamlit web interface for interactive analysis | 
| **FR-4.2** | The system shall support single article text input and analysis | 
| **FR-4.3** | The system shall support CSV file upload for batch processing | 
| **FR-4.4** | The system shall display credibility scores with color-coded indicators | 
| **FR-4.5** | The system shall visualize prediction confidence and text statistics |

### FR-5: Monitoring and Drift Detection

| ID | Requirement |
|----|-------------|
| **FR-5.1** | The system shall collect Prometheus metrics for API requests, latency, and errors. | 
| **FR-5.2** | The system shall monitor prediction distribution and input text characteristics. | 
| **FR-5.3** | The system shall implement drift detection using Population Stability Index (PSI) and Jensen-Shannon divergence. | 
| **FR-5.4** | The system shall compare incoming data against reference statistics from training | 
| **FR-5.5** | The system SHALL flag drift when PSI > 0.25 or JS divergence > 0.05. | 
| **FR-5.6** | The system SHALL expose drift status via GET /drift/status endpoint. |

### FR-6: Model Retraining

| ID | Requirement |
|----|-------------|
| **FR-6.1** | The system shall support model retraining with updated data. | 
| **FR-6.2** | The system shall version retrained models with unique identifiers. | 
| **FR-6.3** | The system shall log retraining events to Weights & Biases when enabled. | 

## Non-functional Requirements

### NFR-1: Performance

| ID | Requirement |
|---|---|
| **NFR-1.1** | Single prediction latency SHALL be <100ms at p95 percentile. |
| **NFR-1.2** | Model loading time SHALL be <5 seconds on system startup. |

### NFR-2: Reliability

| ID | Requirement |
|---|---|
| **NFR-2.1** | System uptime shall be ≥99.5% excluding planned maintenance. |
| **NFR-2.2** | API shall gracefully handle invalid inputs with appropriate error messages |
| **NFR-2.3** | API shall implement retry logic for transient failures. |
| **NFR-2.4** | System shall log all errors with sufficient context for debugging. |
| **NFR-2.5** | API shall provide health check endpoints for monitoring. |

### NFR-3: Scalability

| ID | Requirement |
|---|---|
| **NFR-3.1** | API shall be horizontally scalable via container replication. |
| **NFR-3.2** | System shall support stateless operation (no session affinity required). |
| **NFR-3.3** | Model artifacts shall be loadable from centralized storage (W&B).|

### NFR-4: Reproducibility

| ID | Requirement |
|---|---|
| **NFR-4.1** | All training runs shall use fixed random seeds for determinism. |
| **NFR-4.2** | Preprocessing shall be identical across training and inference. |
| **NFR-4.3** | Model artifacts shall include full configuration and dependency versions. |
| **NFR-4.4** | Experiments shall be logged to Weights & Biases with complete metadata. |
| **NFR-4.5** | Docker images SHALL specify exact library versions (pinned requirements). |

### NFR-5: Maintainability

| ID | Requirement |
|---|---|
| **NFR-5.1** | Code shall follow PEP-8 style guidelines (enforced by Ruff linter). |
| **NFR-5.2** | Functions shall have docstrings explaining parameters and return values. |
| **NFR-5.3** | System shalluse modular architecture with clear separation of concerns. |
| **NFR-5.4** | Configuration shall be externalized (YAML files, environment variables) |
| **NFR-5.5** | Code shall achieve ≥80% test coverage (unit + integration tests). |

### NFR-6: Security 

| ID | Requirement |
|---|---|
| **NFR-6.1** | API shall validate and sanitize all inputs to prevent injection attacks. |
| **NFR-6.2** | System shall not expose sensitive information in error messages or logs. |
| **NFR-6.3** | API keys (W&B) shall be stored in environment variables, never hardcoded. |
| **NFR-6.4** | Docker containers SHALL run as non-root users. |

### NFR-7: Observability

| ID | Requirement |
|---|---|
| **NFR-7.1** | System shall expose Prometheus metrics at /metrics endpoint. |
| **NFR-7.2** | Metrics shall include request count, latency histograms, error rates. |
| **NFR-7.3** | System shall log structured JSON logs with timestamps and log levels. |
| **NFR-7.4** | Grafana dashboards shall visualize key system and model metrics. |
| **NFR-7.5** | Drift detection SHALL generate alerts for Prometheus Alertmanager. |


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

## Technology Selection Justification

### Why Flask for API Framework?

We evaluated several API frameworks before selecting Flask:

| Framework | Pros | Cons | Decision |
|-----------|------|------|----------|
| **Flask** | Simple, mature, easy testing, Prometheus integration, Software Developer is familiar with it | Less features than FastAPI, no async | ✓ **Selected** |
| **FastAPI** | Modern, async, auto-documentation | Learning curve, newer ecosystem | Unnecessary complexity |
| **Django REST** | Full-featured, batteries included | Heavy, overkill for ML serving | Too heavy |

**Rationale**: Flask's simplicity and mature ecosystem make it ideal for our ML model serving. Prometheus integration via `prometheus-flask-exporter` is battle-tested. The fact that our Software Developer is already familiar in Flask allows us to implement the API efficiently without a learning curve.

### Why Prometheus + Grafana for Monitoring?

Prometheus + Grafana was selected because it is the industry-standard open-source stack for cloud-native monitoring. It was chosen over alternative solutions for three main reasons:

- **Cost-Effectiveness**: It provides enterprise-grade monitoring for free, avoiding the high licensing costs of proprietary platforms.
- **Operational Efficiency**: It is lightweight and integrates natively with Docker.
- **Specialized Performance**: It is purpose-built for real-time metrics and alerting, making it the most effective tool for tracking model drift and API health without unnecessary complexity.

### Why Weights & Biases for Experiment Tracking?

Weights & Biases (W&B) was selected for experiment tracking and artifact management due to its seamless integration with Python ML workflows, support for reproducibility, and robust artifact versioning capabilities. It allows us to log training runs, track metrics, and manage model artifacts in a centralized way, which is essential for maintaining traceability and reproducibility across the model lifecycle.

## Risk Analysis
The system is designed for local deployment and reproducible experimentation; the table below summarizes the main risks and the mitigations that are already implemented.

| Risk | Impact | Implemented mitigation |
|---|---|---|
| Data quality / provenance (local CSV vs W&B artifacts; noisy/duplicated text) | Biased training and non-reproducible runs if inputs change | Deterministic preprocessing (`src/data/preprocess.py`); CI uses committed sample dataset (`data/sample_welfake.csv`) |
| Drift (input text and output probability distribution shifts) | Gradual performance degradation and unstable credibility scores | Drift checks in `pipelines/ct_trigger.py` with reports under `results/` and reference stats at `artifacts/reference_stats.json` |
| Manual deployment regressions (wrong config/model alias) | Shipping an unintended model or misconfigured stack | CI builds images and does `docker compose` bring-up with health checks (`.github/workflows/ci-full.yml`) |
| Limited production security / alerting (no auth, no automated alerts) | Unsafe if exposed publicly; delayed incident response | Intended local/offline usage; secrets via env vars must be managed by operators |
| Keyword stuffing / “trigger words” in the input text | Manipulated or unstable credibility scores (false positives/negatives) | Not implemented. Future work: add an input robustness layer (e.g., detect excessive token repetition / abnormal TF-IDF patterns and down-weight or flag the request). |