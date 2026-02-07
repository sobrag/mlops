# Definition of Ready (DoR)
A task can be defined Ready if the objective and expected outcome are clearly defined and agreed upon by the team and if the it's clearly assigned to one (or more) team members.

Acceptance criteria must be specified and aligned with the project’s Definition of Done and dependencies on previous tasks or deliverables must be identified and resolved.
Only Ready items may be pulled into an active sprint.

## Sprint 1
- [ ] Team roles and responsibilities are assigned
- [ ] No prior sprint dependencies exist
- [ ] The dataset source (WELFake) is available and accessible
- [ ] Python environment is configured with libraries
- [ ] Preprocessing scripts for text cleaning and normalization are available
- [ ] The scope of data exploration and preprocessing is clearly defined
- [ ] The expected output of the data pipeline is specified
- [ ] Infrastructure requirements (Docker, repository structure) are agreed upon


## Sprint 2 
- [ ] Cleaned and versioned datasets from Sprint 1 are available
- [ ] Feature representation strategy (e.g., TF-IDF) is agreed upon
- [ ] Baseline model candidates are selected
- [ ] Evaluation metrics and validation strategy are defined
- [ ] Mock inference requirements are specified
- [ ] Monitoring goals for the prototype stage are defined


## Sprint 3
- [ ] A trained and validated model exists
- [ ] Preprocessing logic is finalized and reproducible
- [ ] API input/output contracts are defined
- [ ] Monitoring metrics to be exposed by the real API are identified
- [ ] User Interface scope and interaction flow are agreed upon
- [ ] Deployment and execution environments are available


## Sprint 4
- [ ] Monitoring infrastructure is operational
- [ ] Logged data (inputs and predictions) is available for analysis
- [ ] Drift definitions (what constitutes “drift”) are clearly specified
- [ ] Drift detection techniques and thresholds are selected
- [ ] Simulation strategy for drift scenarios is defined


## Sprint 5
- [ ] Drift signals or simulated drift scenarios are available
- [ ] Retraining or update strategies are defined
- [ ] Evaluation protocol for comparing models is agreed upon
- [ ] Documentation structure and final report outline are prepared
- [ ] Final delivery requirements and submission criteria are known


