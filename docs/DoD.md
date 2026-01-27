# Definition of done

The project is considered done when all the following tasks have been done:

## Sprint 1
- [ ] The WELFake dataset is correctly loaded, cleaned, and documented
- [ ] Relevant columns (title, text, label) are selected and merged
- [ ] Text preprocessing is implemented using deterministic functions (Regex-based)
- [ ] Train/test splits are reproducible and stratified
- [ ] Data preparation code is modular and readable
- [ ] Dataset assumptions and limitations are documented
- [ ] Docker environment is configured (docker-compose.yml created)
- [ ] Project folder structure for code, docs, and monitoring is established

## Sprint 2
- [ ] Baseline model is trained and evaluated
- [ ] Multiple models (LR, SVM, RF, NN, BERT) are implemented and compared
- [ ] Hyperparameter tuning is performed for selected models
- [ ] Performance: Model training achieves a tracked training loss (...)
- [ ] Evaluation metrics include Accuracy, Precision, Recall, F1, ROC
- [ ] The model architecture, parameters, and training process are documented
- [ ] The system can ingest a news article (title and text) and output a credibility score
- [ ] Mock API (Flask) is implemented to simulate inference flow
- [ ] Basic Monitoring Stack (Prometheus & Grafana) is active and collecting metrics from the Mock API

## Sprint 3
- [ ] The real trained model is integrated into the Flask API (replacing the Mock)
- [ ] User Interface is implemented and connected to the API
- [ ] The system supports batch
- [ ] The system can be run end-to-end following documented instructions (tbd)
- [ ] Model predictions can be logged for monitoring purposes
- [ ] A monitoring pipeline is implemented to track input data characteristics and model outputs (credibility score distribution)

## Sprint 4
- [ ] A drift detection method is implemented or simulated and tested 
- [ ] Drift detection thresholds are defined and used to flag potential drift events
- [ ] Drift detection behavior is validated using simulated or time-based data splits
- [ ] The model and preprocessing pipeline are packaged and deployable
- [ ] Model versioning is in place
- [ ] Basic automation exists for retraining or updating the model (after drift detection)
- [ ] The system architecture and ML lifecycle workflow are clearly documented
- [ ] Sprint outcomes are summarized and evaluated


## Sprint 5
- [ ] The model adaptation strategy is defined and executed in response to detected (or simulated) drift
- [ ] At least one model retraining or update is performed using new or temporally restructured data
- [ ] Model performance before and after adaptation is compared using the same evaluation metrics
- [ ] The updated model is verified to: 
* [ ] maintain or improve performance on reference data
* [ ] satisfy the functional requirements of the system
- [ ] The model update process is reproducible, even if manual or semi-automated
- [ ] Retraining decisions (when and why the model is updated) are clearly documented
- [ ] The complete ML lifecycle (train → deploy → monitor → detect drift → update) is described and validated
- [ ] The final system is evaluated against the initial project objectives
- [ ] Limitations, assumptions, and future improvements are clearly stated
- [ ] Final documentation is completed
- [ ] The final report is delivered
