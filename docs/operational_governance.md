# Operational Governance & Versioning

**Project:** News Credibility Estimation
**Version:** 2.0
**Last updated:** 2026‑02‑08
**Status:** Living document – aligned with the current implemented system
**Authors:** Sofia Bragagnolo, Ester De Giosa, Alina Fogar, Riccardo Samaritan
---

## Purpose and Scope

This document defines how the project governs the **end‑to‑end lifecycle of the machine learning system** in production. It focuses on practices that are already implemented in the repository and infrastructure, covering:

* Versioning of **code, data, and models**
* **CI/CD automation** and quality gates
* **Model lifecycle governance**, including reproducibility and experiment tracking
* **Monitoring, drift detection, and incident response**

The objective is to ensure reproducibility, controlled evolution of the system, and operational reliability, without introducing assumptions or mechanisms that are not currently in place.

---

## Version Control Strategy

### Branching Model

* **`main`**: release‑ready branch. All changes are merged via pull requests after passing CI checks.
* **`feature/<topic>`**: short‑lived branches used for incremental development and experimentation.
* **`api`** (legacy): contains a dedicated workflow (`ci-api.yml`) and is only used if explicitly recreated. 
* **`mlops/infrastructure`**: used to stage CI/CD setup, versioning choices, W&B integration, pipelines, and drift-detection scaffolding before merging to `main`.
* feat/ds 

When needed, we open focused topic branches for infrastructure or UI work, following the same PR + CI requirements.

This strategy enforces controlled integration while allowing parallel development.

### Code Versioning

Git is the single source of truth for code.
All changes to `main` are gated by CI checks, including linting and automated tests.

### Model Versioning

Each training run produces a self-contained directory under `artifacts/run_<timestamp>/`.

Stored artifacts include:
- configuration files
- vectorizers
- trained models
- evaluation metrics
- drift reference statistics

A global “active model” pointer or symlink is not maintained; model selection is explicit via configuration.


### Data Versioning

Datasets are **not versioned directly in Git**. The source of truth depends on the execution mode:

* **With W&B enabled** (`use_wandb: true`): datasets are tracked as **W&B Artifacts**, each with a unique version and immutable hash. Pipelines always reference the exact artifact version used during training or evaluation, ensuring full reproducibility.
* **Without W&B** (`use_wandb: false` or offline mode): datasets are loaded from local files under `data/` (e.g. `data/train.csv`, `data/sample_welfake.csv`). These files are **not versioned in Git** and are intended only for local development, testing, or debugging. Reproducibility in this case is limited to the local environment.

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

* **Static analysis**: `ruff check .`
* **Testing**: full `pytest` suite (core logic, API, UI)
* Any failure blocks merging into `main`.

### Training and Drift Smoke Tests

* **Training smoke test**:

  * Executes `pipelines.train_pipeline` with `configs/test.yml`.
  * Verifies that training, serialization, and artifact generation complete successfully.
  * Output is symlinked to `artifacts/models/ci_smoke_latest` for inspection.
* **Drift smoke test**:

  * Runs `pipelines.ct_trigger` on sample data.
  * Ensures that drift detection logic and reporting paths remain functional.

### Containerization Checks

* Docker images are built for:

  * Inference API (`src/app/Dockerfile`)
  * Streamlit UI (`src/ui/Dockerfile`)
* `docker-compose.yml` is validated in CI.
* Basic health checks are executed for API, Prometheus, Grafana, and Streamlit services.

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
* ****Aliases****: each training run logs aliases `latest` (moving) and `run/<run_id>` (immutable); promotion aliases like `staging` or `production` can be set via `scripts/promote_model.py`.

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

### Incident Response | Trigger | Action | Outcome |
|-------|--------|---------|
| CI failure | Inspect logs, fix code or configuration, re-run pipeline | Restored passing CI |
| Drift alert | Analyze `drift_report.json`, assess feature shifts, decide on retraining | Updated or confirmed model |
| Service outage | Restart docker-compose stack, verify `/health`, inspect container logs | Service availability restored |
| Quality regression | Re-evaluate on held-out data and compare with previous metrics | Decision to rollback or retrain |

**Post-incident actions** include documenting the root cause and updating configuration,
preprocessing steps, or thresholds when necessary.

### Monitoring Stack

* **Prometheus** and **Grafana** are included via `docker-compose.yml` with default provisioning.
* Service availability is verified in CI using HTTP checks.
* Logging:

  * Structured Python logs from pipelines
  * Container logs via `docker compose logs`
* Custom application‑level Prometheus metrics are not yet implemented.

### Docker setup

* Local bring‑up: `docker compose up -d` (API, Streamlit, Prometheus, Grafana).
* Tear‑down: `docker compose down -v`.
* Prereqs: model artifacts present under `artifacts/` (or mounted), dataset in `data/WELFake_dataset.csv`, and default ports 5001/8501/9090/3000 available.

### Testing strategy

* **Test runner**: `pytest`.
* **Suites**:
  * Core pipeline tests under `tests/` (data/preprocess/model logic).
  * API tests under `tests/api/`.
  * UI tests under `tests/ui/`.
* **Commands**:
  * All tests: `python -m pytest tests/ -v --ignore=tests/api/ --ignore=tests/ui/` (core); `python -m pytest tests/api/ -v --tb=short`; `python -m pytest tests/ui/ -v --tb=short`.
* **CI**: executes the three suites plus lint (`ruff check .`) in `.github/workflows/ci-full.yml`.

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
