# Operational Governance & Versioning Document

## News Credibility Estimation

**Version:** 2.0
**Last updated:** 2026‑02‑08

**Team:**
- **Sofia Bragagnolo** – Project Manager  
- **Ester De Giosa** – Data Scientist  
- **Alina Fogar** – MLOps Engineer  
- **Riccardo Samaritan** – Software Developer  

---

## Table of Contents
- [Operational Governance \& Versioning Document](#operational-governance--versioning-document)
  - [News Credibility Estimation](#news-credibility-estimation)
  - [Table of Contents](#table-of-contents)
  - [Purpose and Scope](#purpose-and-scope)
  - [Version Control Strategy](#version-control-strategy)
    - [System and Platform](#system-and-platform)
    - [Branching Model](#branching-model)
    - [Code Versioning](#code-versioning)
    - [Model Versioning](#model-versioning)
    - [Data Versioning](#data-versioning)
  - [CI/CD and Automation](#cicd-and-automation)
    - [Continuous Integration](#continuous-integration)
    - [Quality Gates](#quality-gates)
    - [Deployment](#deployment)
  - [Model Lifecycle Governance](#model-lifecycle-governance)
    - [Model Registry](#model-registry)
    - [Reproducibility](#reproducibility)
    - [Experiment Tracking](#experiment-tracking)
  - [Monitoring and Maintenance](#monitoring-and-maintenance)
    - [Drift Detection](#drift-detection)
    - [Incident Response](#incident-response)
    - [Monitoring Stack](#monitoring-stack)
      - [Components](#components)
      - [Metrics Collected](#metrics-collected)
    - [Docker setup](#docker-setup)
  - [Runtime and Environment Governance](#runtime-and-environment-governance)
  - [Containerization Policy](#containerization-policy)
  - [Inference Interface and API Governance](#inference-interface-and-api-governance)
  - [Configuration Governance](#configuration-governance)
  - [Scope and Limitations](#scope-and-limitations)
  - [Closing Notes](#closing-notes)

## Purpose and Scope

This document defines the **end‑to‑end operational governance** of the News Credibility Estimation system, covering how we manage the complete ML lifecycle from development through production deployment. 

It establishes practices and procedures that ensure:

* **Reproducibility**: Every experiment, training run, and deployment can be exactly replicated.
* **Traceability**: Full audit trail from code commits to model predictions.
* **Reliability**: Controlled changes, automated testing, and monitoring prevent production incidents.

The governance model focuses on these pillars:

* Versioning of **artifacts** (code, data, and models) to track changes and ensure reproducibility.
* **CI/CD automation** and quality gates.
* **Model lifecycle governance**, including reproducibility and experiment tracking
* **Monitoring, drift detection, and incident response** to maintain system health.

---

## Version Control Strategy

Version control is a core governance mechanism for this project, as it defines how changes are proposed, reviewed, validated, and integrated across the system lifecycle.

We adopted a **trunk-based development** model with feature branches to balance collaboration speed with stability.

### System and Platform

| Aspect | Details |
| --- | --- |
| Version Control System | Git |
| Hosting Platform | GitHub |
| Repository | [https://github.com/sobrag/mlops](https://github.com/sobrag/mlops) |

### Branching Model

* **`main`**: release‑ready branch. All changes are merged via pull requests after passing CI checks.
* **`feature/<topic>`**: short‑lived branches used for incremental development and experimentation.
* **`api`**: contains a dedicated workflow (`ci-api.yml`) and is only used if explicitly recreated. 
* **`mlops/infrastructure`**: used to stage CI/CD setup, versioning choices, W&B integration, pipelines, and drift-detection scaffolding.
* **`mlops/streamlit-ui`**: used to stage Streamlit UI development.

When needed, we opened focused topic branches for infrastructure or UI work, following the same PR + CI requirements.

This strategy enforces controlled integration while allowing parallel development, and avoids team members stepping on each other's work or bottlenecks in the development process.

### Code Versioning

Git is the single source of truth for code.
All changes to `main` are gated by CI checks, including linting and automated tests.

### Model Versioning

Git is designed for text files, not large binary models. Committing model weights (50+ MB) bloats repository history.

To solve this problem, we separated model artifacts from code and use a structured artifact management for versioning. Models are stored as files under `artifacts/` and tracked via **Weights & Biases Artifacts** when enabled.

Each training run produces a **model bundle** containing:

```
artifacts/run_20260209_143052/
├── model.pkl              # Trained Logistic Regression (joblib)
├── vectorizer.pkl         # Fitted TF-IDF vectorizer (joblib)
├── config.yml             # Training configuration (hyperparams, data splits)
├── metrics.json           # Evaluation results (accuracy, F1, etc.)
└── training_log.txt       # Training process logs
```

No local ‘active model’ symlink is maintained; model resolution is configuration-driven (**W&B** alias) or auto-resolved from the latest local artifacts directory.

**Versioning strategy**:

1. **Local Storage**:
   - Every training run saves artifacts to `artifacts/run_<timestamp>/`
   - `.gitignore` excludes `artifacts/` directory (not committed to Git)
   - Developers maintain local history for recent experiments

2. **Weights & Biases Artifacts**:
   - Each training run logs artifacts to W&B with unique version
   - Artifact name: `credibility_model_bundle`
   - Version format: `v<sequential_number>` (auto-incremented)
   - Hash: SHA-256 checksum for integrity verification

3. **Aliases** (Model Promotion):
   - `latest`: Most recent training run (moving pointer)
   - `run/<run_id>`: Immutable pointer to specific W&B run
   - `staging`: Candidate for production (set manually)
   - `production`: Currently deployed model (set manually)

**Example W&B Artifact Timeline**:

| Version | Alias | Accuracy | Timestamp | Status |
|---------|-------|----------|-----------|--------|
| v1 | | 0.905 | 2025-12-15 | Archived |
| v2 | | 0.918 | 2025-12-20 | Archived |
| v3 | staging | 0.923 | 2026-01-10 | Candidate |
| v4 | latest, production | 0.925 | 2026-02-05 | Deployed |

### Data Versioning

Datasets are handled with a split strategy to balance reproducibility and repository size. The source of truth depends on the execution mode:

* **With W&B enabled** (`use_wandb: true`): datasets are tracked as **W&B Artifacts**, each with a unique version and immutable hash. Pipelines always reference the exact artifact version used during training or evaluation, ensuring full reproducibility.
* **Without W&B** (`use_wandb: false` or offline mode): datasets are loaded from local files under `data/` (e.g. `data/sample_welfake.csv`, `data/synthetic_inference_sample.csv`, incoming batches under `data/incoming/`).

  * Small, curated **sample datasets are committed to Git** to support local development and CI reproducibility.
  * The full WELFake dataset is **not committed to Git**; it is expected to be provided locally (e.g. `data/WELFake_dataset.csv`) or via W&B Artifacts when enabled.

This separation avoids committing large or sensitive datasets to the repository, while still allowing reproducible experiments when artifact tracking is enabled.

---

## CI/CD and Automation

**Platform**: GitHub Actions

### Continuous Integration

* The main workflow (`.github/workflows/ci-full.yml`) is triggered on:

  * `push` and `pull_request` events targeting `main`
  * Manual `workflow_dispatch`
* A concurrency guard prevents overlapping runs for the same branch or pull request.

### Quality Gates

All checks below are implemented in `.github/workflows/ci-full.yml`. A failure in any check blocks merging into `main`.

| Check | What it runs | Purpose |
| --- | --- | --- |
| Lint | `ruff check .` | Enforce style and catch static issues |
| Tests (core ML) | `python -m pytest tests/ -v --ignore=tests/api/ --ignore=tests/ui/` | Validate preprocessing, training utilities, artifact I/O |
| Tests (API) | `python -m pytest tests/api/ -v --tb=short` | Validate Flask endpoints and service integration (mock/real artifacts) |
| Tests (UI) | `python -m pytest tests/ui/ -v --tb=short` | Validate Streamlit client logic and response handling |
| Smoke train | `python -m pipelines.train_pipeline --config configs/test.yml --max-rows 200` | Ensure training pipeline executes end-to-end and produces artifacts |
| Smoke drift | `python -m pipelines.ct_trigger --config configs/ct.yml --incoming data/sample_welfake.csv` (W&B disabled) | Ensure drift pipeline executes and writes a report |
| Docker compose validation | `docker compose config --quiet` | Validate `docker-compose.yml` is syntactically correct |
| Docker image builds | `docker build` for API and Streamlit | Ensure Dockerfiles build |
| Compose bring-up + health | `docker compose up -d` + `curl` checks | Verify services start and key endpoints respond |
| Cleanup | `docker compose down -v` | Ensure CI job is self-cleaning |


### Deployment

* Deployment is **manual** after a green CI run.
* The docker‑compose stack is restarted using the latest validated artifacts.
* No automated model alias promotion or rollback mechanism is implemented.

---

## Model Lifecycle Governance

### Model Registry

* **Backend**: Weights & Biases Artifacts.
* **Model bundle name**: `credibility_model_bundle`.
* Inference resolves:

  * Either a local artifact directory under `artifacts/`, or
  * A W&B artifact specified via configuration (`model_artifact`).
  * ****Aliases****: each training run logs alias `latest` (moving) and a run-specific immutable alias (pattern depends on the pipeline implementation). Promotion aliases like `staging` or `production` are set manually in W&B when used.

### Reproducibility

* Fixed random seed (`seed: 42`) ensures deterministic behavior where applicable.
* Preprocessing is centralized in `src/data/preprocess.py`.
* Each training run stores:

  * Training configuration (YAML)
  * TF‑IDF vectorizer
  * Model parameters
  * Evaluation metrics
  * (Drift reference statistics are generated by the drift pipeline, not during training.)
* Dataset provenance is captured via configuration (`dataset_artifact`).
* Inference and drift pipelines persist JSON summaries under `results/` for auditability.

### Experiment Tracking

* **Tool**: Weights & Biases.
* Tracking can be enabled or disabled via `use_wandb` and environment variables (`WANDB_MODE`, `USE_WANDB`).
* Logged information includes:

  * Hyperparameters and data splits
  * Training and evaluation metrics
  * Drift baselines and reports
  * Model and dataset artifacts
* Training, inference, and drift pipelines all emit structured logs and artifacts for traceability.

---

## Monitoring and Maintenance

### Drift Detection

* Implemented in `pipelines.ct_trigger.py` using `configs/ct.yml`.
* **Reference window**: first 70% of the dataset.
* **Evaluation window**: tail batches (typically 80–100%).
* **Metrics and thresholds**:

  * Population Stability Index (PSI):

    * Text length distribution
    * Prediction distribution
    * Threshold: `0.25`
  * Jensen–Shannon divergence on token frequencies:

    * Threshold: `0.05`
* Drift is flagged if any threshold is exceeded.
* Outputs:

  * Reference statistics: `artifacts/reference_stats.json`
  * Batch reports: `results/drift/<timestamp>/drift_report.json`
  * Optional W&B artifact and alert
* **Auto‑retraining**:

  * Controlled by `auto_retrain` flag (default: `false`).
  * When enabled, retraining is triggered using `retrain_config`.

### Incident Response

| Trigger | Action | Outcome |
| --- | --- | --- |
| CI failure | Inspect logs, fix code or configuration, re-run pipeline | Restored passing CI |
| Drift alert | Analyze `drift_report.json`, assess feature shifts, decide on retraining | Updated or confirmed model |
| Service outage | Restart docker-compose stack, verify `/health`, inspect container logs | Service availability restored |
| Quality regression | Re-evaluate on held-out data and compare with previous metrics | Decision to rollback or retrain |


**Post-incident actions** include documenting the root cause and updating configuration,
preprocessing steps, or thresholds when necessary.

### Monitoring Stack

#### Components

1. **Prometheus**: Metrics collection and storage
2. **Grafana**: Visualization and alerting
3. **Flask Prometheus Exporter**: Instrument API

Both Prometheus and Grafana are included in the `docker-compose.yml` stack with default provisioning. The API is instrumented to expose metrics at `/metrics` using the Flask Prometheus Exporter.

#### Metrics Collected

**System Metrics**:
| Metric | Type | Description |
|--------|------|-------------|
| `flask_http_request_total` | Counter | Total API requests (by endpoint, status) |
| `flask_http_request_duration_seconds` | Histogram | Request latency distribution (p50, p95, p99) |
| `flask_http_request_exceptions_total` | Counter | Exception count by type |

**Model Metrics**:
| Metric | Type | Description |
|--------|------|-------------|
| `drift_is_drifted` | Gauge | Binary flag: 0 (no drift), 1 (drift detected) |
| `drift_sample_count` | Gauge | Number of predictions in drift detection window |
| `drift_prediction_mean_shift` | Gauge | Shift in average credibility score |
| `drift_text_length_shift` | Gauge | Shift in average input text length |

System metrics are collected continuously while the API is running, while model metrics are updated after every prediction.

### Docker setup

* Local bring‑up: `docker compose up -d` (API, Streamlit, Prometheus, Grafana).
* Tear‑down: `docker compose down -v`.
* Prereqs: model artifacts present under `artifacts/` (or mounted), dataset in `data/WELFake_dataset.csv`, and default ports 5001/8501/9090/3000 available.

---

## Runtime and Environment Governance

The system is designed to run in a controlled and reproducible environment.
Execution is supported both in local development setups and in CI contexts.

The runtime environment is governed by:
- a fixed Python version and pinned dependencies
- optional containerization to standardize execution
- explicit configuration files defining execution modes

Environment-specific behavior (e.g. local vs CI, online vs offline execution) is controlled via configuration flags and environment variables.

## Containerization Policy

Docker is used to provide a reproducible execution environment and to reduce dependency on local system configurations.

Containers are intended for:
- local development
- integration testing
- monitoring stack execution (where applicable)

Automated image publishing, orchestration, and production-grade container deployment are explicitly out of scope for this project.

## Inference Interface and API Governance

Model inference is exposed through a lightweight interface, intended primarily for local and offline usage.

The inference interface is governed by the following principles:
- inference is explicitly triggered (script- or API-based)
- model selection is configuration-driven
- no persistent, externally managed API service is assumed

Authentication, authorization, and API gateway management are not addressed in the current scope.

## Configuration Governance

All operational behavior is controlled via explicit configuration files and environment variables.

Configuration files define:
- dataset and model references
- execution modes (training, inference, drift detection)
- thresholds and operational parameters

Environment variables are used only for environment-specific or sensitive settings.
Configuration files are versioned in Git, while secrets are never committed.

## Scope and Limitations

This operational governance focuses on reproducibility, traceability, and controlled execution of ML pipelines.

The following aspects are explicitly out of scope:
- automated deployment to production environments
- managed API gateways and authentication layers
- cloud-native orchestration platforms
- service-level objectives (SLOs) and uptime guarantees

These limitations are intentional and aligned with the goals of the project.

## Closing Notes

This governance setup prioritizes **clarity, reproducibility, and controlled change**, while remaining aligned with what is concretely implemented in the project. Future extensions (e.g. automated promotion, advanced metrics, or policy‑driven governance) can be layered on top without invalidating the current structure.
