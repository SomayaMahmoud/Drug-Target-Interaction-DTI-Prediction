DTI Prediction Pipeline: AI-Driven Drug-Target Interaction SystemThis project is an integrated platform leveraging Artificial Intelligence to predict Drug-Target Interaction (DTI). Designed collaboratively, it combines traditional machine learning and deep learning approaches to accelerate bioinformatics and early-stage drug discovery. 

🚀 OverviewThe primary goal of this system is to reduce the time and cost associated with early drug discovery phases by accurately predicting the binding affinity between specific drug molecules and target proteins using advanced machine learning models.

🛠 Key FeaturesSmart Data Processing: Supports popular drug discovery datasets (Davis, KIBA) with advanced data cleaning and chemical structure validation.  Comprehensive Feature Engineering: Extracts molecular representations using Morgan Fingerprints (ECFP4), physicochemical properties (molecular descriptors), and amino acid compositions.  Ensemble Approach (Multiple Architectures):Classical ML Models: Ridge Regression, Random Forest, and XGBoost.  Deep Learning Models: Custom deep LSTM networks and PyTorch MLPs featuring batch normalization and dropout.  Drug Recommendation System: Enables querying a target protein against a drug library to rank and recommend the top candidate binders.  Interactive UI: Built-in Gradio web interface for seamless model testing and visualization.  Explainable AI & Evaluation: Utilizes comprehensive metrics (MSE, RMSE, $R^2$, Concordance Index, ROC-AUC), SHAP analysis, feature importance plots, and error analysis.

🏗 System Architecturedata.py: Data loading, cleaning, and exploratory data analysis.  features.py: Feature engineering, scaling, and handling class imbalances.  models_classical.py & models_deep.py: Classical machine learning and deep learning training pipelines.  evaluate.py: Performance evaluation, metrics calculation, and visualization.  recommend.py: Drug recommendation engine.  main.py: Main orchestration script running the entire pipeline.  

📦 RequirementsThe project relies on core data science and bioinformatics libraries:
rdkit, numpy, pandas, scikit-learn, xgboost, torch, tensorflow, gradio.  You can install all dependencies via:Bashpip install -r requirements.txt

💡 Getting StartedEnsure your dataset files (davis_all.csv, kiba_all.csv) are placed in the project directory.  Run the complete pipeline using:Bashpython main.py
Evaluation metrics, generated plots, and the interactive Gradio UI will launch automatically.
