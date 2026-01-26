# PROJECT PROPOSAL & DEVELOPMENT PLAN

**Project Title:** News Credibility Estimation with Drift Detection and Model Monitoring  

**Course Context:** MLOps  

**Roles:**
- **Sofia Bragagnolo** – Project Manager  
- **Ester De Giosa** – Data Scientist  
- **Alina Fogar** – MLOps Engineer  
- **Riccardo Samaritan** – Software Developer  

The team's main communication tool was Slack.

---

## PROJECT SUMMARY

### Scope
The goal of this project is to design and implement a **complete MLOps pipeline** for estimating the credibility of news articles.  
The project does not focus solely on model training, but on the **entire machine learning lifecycle**.

Instead of producing a binary classification (real vs fake), the system outputs a **credibility score**, representing the degree of reliability of a news article based on its textual characteristics.  

An important aspect of the project is the management of **changes over time**, addressing data drift and model performance degradation caused by the evolution of language, different topics, and writing styles in news content.

---

### Objectives

First of all, we want to develop a machine learning model capable of estimating news credibility from textual data. The idea is to build a **reproducible and automated ML pipeline**, covering data preparation, training, evaluation, and deployment.
We want to implement **model monitoring** to observe predictions and system behavior in production and detect **data drift and concept drift**.
We also want to support **model retraining and system adaptation** when monitored metrics indicate degradation.

---

### Relevance
This project addresses the challenge of **misinformation**, highlighting how machine learning systems can assist users in evaluating the reliability of online content.
Buisness relevance: to define monetization of the problem

---

## DELIVERABLES

### Data Pipeline
The data pipeline includes:
- Data ingestion and validation
- Text preprocessing: text cleaning using Regex
- Feature extraction via TF-IDF for classical models
- Dataset versioning
- Reproducible train/validation/test splits

### ML Kernel (Prediction Model)

- Supervised learning model trained on labeled news articles. Model Selection:
* Baselines: Logistic Regression, Support Vector Machines (SVM), and Random Forest.
* Ensemble Methods: VotingClassifier to combine multiple model predictions.
* Advanced Model: BERT (BertForSequenceClassification) for deep linguistic understanding.

- Conversion of classification outputs into a **continuous credibility score**: The system uses precision_recall_fscore_support and roc_curve to analyze the binary classification outputs before converting them into a continuous credibility score.

- Offline evaluation using appropriate performance metrics


### CI/CD Pipeline
To bridge development and production:
- Automated training and testing workflows
- Versioning of models and artifacts
- Controlled deployment of new model versions

### Monitoring and Drift Detection

---

## MILESTONES
The objectives of the sprints we defined in the project proposal: 
1. Data exploration and preprocessing completed  
2. Baseline credibility estimation model developed  
3. Automated pipeline implemented  
4. Monitoring and drift detection enabled  
5. System adaptation and final validation completed  

---

## WORK BREAKDOWN STRUCTURE (WBS)

### 1. Project and Data Management (sprint 1)
- Requirements definition
- Define project scope and objectives
- Sprint planning and coordination
- Progress tracking and risk management
- Documentation and reporting
- Final delivery coordination

- Dataset acquisition and exploration (WELFake)
- Data quality assessment (missing values, duplicates, noise)
- Data understanding and preparation
- Text preprocessing (cleaning, normalization)
- Feature engineering
- Dataset splitting and versioning

### 2. Credibility Estimation Model Development (sprint 2)
- Baseline model selection and implementation
- Model training and hyperparameter tuning
- Model evaluation and metric selection
- Credibility score definition and calibration
- Model documentation

### 3. Deployment and System Integration (sprint 3)
- Design system architecture
- Implement inference pipeline
- Develop input/output interfaces
- Integrate model with preprocessing pipeline
- end-to-end workflow testing
- Pipeline automation

### 4. Monitoring and Drift Observation (sprint 4)
- Define monitoring metrics for inputs and outputs
- Implement data and/or prediction drift detection methods
- Simulate data drift using time-based data splits?
- Validate drift detection results
- Define retraining strategy 

### 5. System Adaptation and Finalization (sprint 5)
- Implement model update workflow
- Re-evaluate updated model performance
- Final system validation
- Performance comparison before and after drift

---

## GANTT-STYLE SCHEDULE

### **Sprint 1 – Data Understanding and Preparation**

* Exploratory analysis of the WELFake dataset
* Data cleaning and text preprocessing
* Definition of the initial data processing pipeline
* Setup of project structure and documentation

**Deliverables:**

* Cleaned and preprocessed dataset
* Documented data pipeline
* Initial project documentation

---

### **Sprint 2 – Credibility Estimation Model Development**

* Selection and implementation of a baseline model
* Model training and hyperparameter tuning
* Definition of evaluation metrics
* Design and calibration of the credibility score

**Deliverables:**

* Trained baseline credibility model
* Evaluation results and metric analysis
* Defined credibility scoring approach

---

### **Sprint 3 – System Integration**

* Integration of preprocessing and model components
* Model deployment setup
* Implementation of the inference workflow
* Automation of the training and inference pipeline

**Deliverables:**

* End-to-end executable system
* Deployed model with inference capability

---

### **Sprint 4 – Monitoring and Drift Detection**

* Definition and implementation of monitoring metrics
* Monitoring of input data and model predictions
* Implementation of at least one drift detection method
* Analysis of system behavior over simulated time

**Deliverables:**

* Monitoring and logging components
* Drift detection mechanism
* Drift analysis results

---

### **Sprint 5 – System Adaptation and Finalization**

* Definition of a model retraining or update strategy
* Model adaptation based on detected drift
* Final system validation and performance assessment
* Final documentation and project delivery

**Deliverables:**

* Updated or retrained model
* Final validated system
* Complete project documentation and final report

---

## DEFINITION OF DONE (DoD)

## DEFINITION OF READY (DoR)

---

## RESOURCES & INFRASTRUCTURE

- **Dataset:** WELFake  
- **Programming Language:** Python 3.12  
- **ML & Data Libraries:** 
* Data Handling: pandas, numpy, datasets (Hugging Face)
* Visualization: matplotlib, seaborn, wordcloud
* Preprocessing: re (Regular Expressions), TfidfVectorizer
* Classical ML (Scikit-learn): LogisticRegression, SVC, LinearSVC, RandomForestClassifier, MLPClassifier, KMeans, VotingClassifier
* Deep Learning & NLP: torch (PyTorch), transformers (BERT Tokenizer and Sequence Classification)
- **MLOps Tools:** Git, CI/CD pipelines, monitoring frameworks 
- **Resources:** 
- **Infrastructure:** 

---

## THEORETICAL REFERENCES
- Treveil, M. et al., *Introducing MLOps*, O’Reilly Media  