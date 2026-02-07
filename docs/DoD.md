# Definition of done

The project is considered done when all the following tasks have been done:

## Sprint 1
- [ ] The WELFake dataset is correctly loaded, cleaned, and documented
- [ ] Relevant columns (title, text, label) are selected and merged into a unified textual representation
- [ ] Deterministic text preprocessing (regex-based normalization and cleaning) is implemented and reproducible
- [ ] Dataset splits (train/validation/test) are reproducible and stratified by label
- [ ] Data preparation code is modular and readable
- [ ] Dataset assumptions, limitations, and potential biases are explicitly documented
- [ ] A Docker-based development environment is configured (docker-compose.yml) and executable
- [ ] A clear and consistent project folder structure for code, data, documentation, and monitoring is established

## Sprint 2
- [ ] The Baseline credibility model is trained and evaluated
- [ ] Multiple models are implemented and compared
- [ ] Hyperparameter tuning is performed for selected models
- [ ] Model evaluation includes both classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) and probability-based analysis (where applicable)
- [ ] A credibility score is clearly defined and derived from model outputs
- [ ] The system can ingest a news article (title and text) and return a credibility score
- [ ] A Mock inference API (Flask) is implemented to simulate production inference flow
- [ ] A basic monitoring stack (Prometheus and Grafana) is deployed and collects metrics from the Mock API
- [ ] The model architecture, parameters, and training process are documented

## Sprint 3
- [ ] The real trained model is integrated into the Flask API (replacing the Mock)
- [ ] Vectorizer, model, and calibration artifacts are correctly loaded and used
- [ ] User Interface (Streamlit dashboard) is implemented and connected to the API
- [ ] The system supports batch workflow
- [ ] The system can be run end-to-end (data → model → API → UI) following documented instructions
- [ ] Model predictions can be logged for monitoring purposes
- [ ] A monitoring pipeline is implemented to track input data characteristics and model outputs (credibility score distribution)

## Sprint 4
- [ ] Monitoring metrics for system behavior, inputs, and outputs are clearly defined and implemented
- [ ] A drift detection method (statistical or simulated) is implemented and tested 
- [ ] Drift thresholds are defined and used to flag potential drift events
- [ ] Drift behavior is validated using simulated or time-based data splits
- [ ] The model and preprocessing pipeline are versioned and deployable
- [ ] A basic retraining or update workflow is defined and executable (after drift detection) 
- [ ] The system architecture and ML lifecycle workflow are clearly documented
- [ ] Sprint outcomes are summarized and evaluated against monitoring objectives


## Sprint 5
- [ ] The model adaptation strategy is defined and executed in response to detected (or simulated) drift
- [ ] Model retraining or update is performed using new or temporally restructured data
- [ ] Model performance before and after adaptation is compared using the same evaluation metrics
- [ ] The updated model: 
  * [ ] maintains or improves performance on reference data
  * [ ] satisfies the functional requirements of the system
- [ ] The model update process is reproducible
- [ ] Retraining decisions (when and why the model is updated) are clearly justified and documented
- [ ] The complete ML lifecycle (train → deploy → monitor → detect drift → update) is described and validated
- [ ] The final system is evaluated against the initial project objectives
- [ ] Limitations, assumptions, and future improvements are clearly stated
- [ ] Final documentation is completed