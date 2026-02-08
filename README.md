# News Credibility Checker

## Machine Learning Operations Final Project

The proliferation of misinformation degrades public trust and negatively impacts individual and collective decision-making. The business objective of this project is to support users and digital platforms by providing an assessment of the credibility of news articles, helping to reduce the risk of spreading unreliable information.

To this end, the project focuses on the design and implementation of a system for estimating news credibility. Rather than making a strict true/false judgment, the system assigns a credibility score that reflects how reliable an article appears based on its textual characteristics.

To address the dynamic nature of news content, since language and writing styles evolve over time, the system is designed to observe changes in incoming data and to support updates when its behavior no longer reflects current information.

## Features

- **Credibility Scoring**: Assigns a 0-100 credibility score to news articles
- **REST API**: Flask-based API for single and batch predictions
- **Web Interface**: Streamlit UI for interactive analysis
- **Model Versioning**: Weights & Biases integration for model and dataset tracking
- **Drift Detection**: Automated monitoring for data and concept drift
- **Monitoring**: Prometheus metrics with Grafana dashboards

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   Flask API     │────▶│   ML Model      │
│   (port 8501)   │     │   (port 5001)   │     │   (TF-IDF + LR) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   Prometheus    │────▶ Grafana (port 3000)
                        │   (port 9090)   │
                        └─────────────────┘
```

## Tech Stack

- **ML**: scikit-learn, TF-IDF vectorization, Logistic Regression
- **API**: Flask, Gunicorn
- **UI**: Streamlit
- **MLOps**: Weights & Biases (experiment tracking, model registry)
- **Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) Weights & Biases account for model download

### Running with Docker

1. **Clone the repository**
   ```bash
   git clone https://github.com/sobrag/mlops.git
   cd mlops
   ```

2. **Set up W&B credentials** (to download the production model)
   ```bash
   export WANDB_API_KEY=your_api_key_here
   ```

3. **Start all services**
   ```bash
   docker compose up --build -d
   ```

4. **Access the applications**
   - **Web UI**: http://localhost:8501
   - **API**: http://localhost:5001
   - **Prometheus**: http://localhost:9090
   - **Grafana**: http://localhost:3000 (admin/admin)

### Running Locally (Development)

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Login to W&B** (optional, for model download)
   ```bash
   wandb login
   ```

3. **Train a model** (if no model artifacts exist)
   ```bash
   python pipelines/train_pipeline.py --config configs/train.yml
   ```

4. **Run the API**
   ```bash
   python -m src.app.main
   ```

5. **Run the UI** (in another terminal)
   ```bash
   streamlit run src/ui/app.py
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single text prediction |
| `/predict/batch` | POST | Batch predictions |
| `/metrics` | GET | Prometheus metrics |

### Example Request

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here..."}'
```

### Example Response

```json
{
  "credibility_score": 75.5,
  "probability": 0.755,
  "label": "real"
}
```

## Project Structure

```
mlops/
├── configs/              # Training and CT configurations
├── data/                 # Sample datasets
├── docs/                 # Documentation
├── pipelines/            # Training and inference pipelines
├── src/
│   ├── app/              # Flask API
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature extraction (TF-IDF)
│   ├── models/           # Model training and prediction
│   ├── monitoring/       # Prometheus/Grafana configs
│   ├── ui/               # Streamlit interface
│   └── utils/            # Utilities and artifacts management
└── tests/                # Unit and integration tests
```

## Dataset

The model is trained on the [WELFake dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification), which contains ~72,000 news articles labeled as real (0) or fake (1).
