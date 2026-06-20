# Predictive Maintenance using Machine Learning

## Project Objective
Predict machine failure using machine sensor data before breakdown occurs.

## Dataset
AI4I Predictive Maintenance Dataset

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- SMOTE
- Joblib
- Streamlit

## Workflow
- Data Preprocessing
- Exploratory Data Analysis
- Feature Correlation Analysis
- Data Cleaning
- Label Encoding
- Train/Test Split
- SMOTE Balancing
- Random Forest Training
- Model Evaluation
- Model Deployment using Streamlit

## Model Performance
Accuracy: 92.4%

Classification Report:

Class 0 (No Failure)
- Precision: 98%
- Recall: 94%

Class 1 (Failure)
- Precision: 47%
- Recall: 76%

## Run Project

pip install -r requirements.txt

streamlit run app.py