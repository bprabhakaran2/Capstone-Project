# Capstone Project - Medication Recommendation Prediction
## Project Overview
This project demonstrates the step-by-step process of finding the best model that can be utilized to recommend a medication based on condition, symptom and other demographic information for a patient. The dataset used to perform this analysis is sourced from https://www.kaggle.com/datasets/asjad99/mimiciii/data, the data is further grouped and synthetic data is added to create a reasonable pool of records to analyze and train a model.

MIMIC-III (Medical Information Mart for Intensive Care) is a freely accessible database developed by the MIT Lab for Computational Physiology. It contains detailed information about over 60,000 ICU admissions to Beth Israel Deaconess Medical Center between 2001 and 2012. For this analysis, the data is derived from Patient, Prescriptions, Admissions , ICD and Drug code datasets.

Building and implementing a predictive model for medication recommendation in an EHR system will enable providers to make informed medication decisions and will prove to be a useful tool to study medication effectiveness and additionally will help minimize hospitalization and provider visits.

Disclaimer:This project is a preliminary effort to predict with a limited dataset and is would need additional preprocessing for larger datasets or neural network based models to integrate with an EHR system.

The input dataset includes following columns derived from MIMIC-III along with synthetic data augmented:

Columns - Description
PatientID - Incremental value masked
Age - Age
Gender - Gender M/F
BMI - Body Mass Index
Weight_kg - Weight
Height_cm - Height
Chronic_ Conditions - Existing condition
Symptoms - Symptoms
Diagnosis - Diagnosis code
Recommended_ Medication - Medication prescribed for existing condition/symptom
NDC - National Drug Code
Dosage - Medication Dosage
Duration - Duration for Medication
Treatmen_ Effectiveness - Effectiveness of Medication
Adverse_ Reactions - Any allergic reactions or side-effects
Recovery_ Time_Days - Recovery time average

## Project Workflow 
1. Data load and preprocessing               
2. Exploratory Data Analysis               
3. Feature Engineering and Preprocessing                  
4. Split Train/test                 
5. Baseline Performance Analysis                  
6. Class Balancing                    
7. Simple Model Train - LogisticRegression                 
8. Classification Model Training on imbalanced and balanced sets: DecisionTree, RandomForest, LightGBM, XGBOOST
9. Model Performance Comparison
10. Model Evaluation on test dataset
11. Cross-validation evaluation
12. Hyperparameter tuning using RandomizedSearchCV
13. Model Selection and Feature Importance

## NoteBooks
1. dataaugmentation.ipynb : Data extraction and grouping from MIMIC III dataset, augment additional features such as Duration, Adverse_Reactions, Treatment_Effectiveness, Recovery_Time_Days
2. MedicationRecommedation.ipynb : Data exploration, Feature engineering, Model Training, Hyperparameter Tuning, Model Evaluation and Selection

## Key Features

1. Age
2. BMI
3. Chronic_Condition
4. Symptoms
5. Gender
6. Diagnosis(ICD Code)
7. NDC (Prescription Drug code)

## Conclusion

### Findings

1. Simple regression model did not provide expected outcome , this is expected since this is a mutilabel classification problem
2. Text columns chronic condition and symptom had several clinical note shorthands that requires mapping to readable words
3. LightGBM outperformed other classification models with better Train and Test F1 score indicating good generalization
4. Key features such as Chronic conditions(wound,carotid,atrial,regurgitation,exacerbation,pancreatitis) , Weight, Height , Gender , BMI , symptoms were selected as top features for most of the labels

### Recommendations and Future Work
1. Utilization of Word2Vec and mapping for clinical notes shortforms in symptoms               
2. Exploring Graph Neural Network and deep learning for better prediction

## Tech Stack

Python: pandas, NumPy, scikit-learn, SHAP             
Data Visualization: Matplotlib, Seaborn, plotly                    
Machine Learning Techniques: Logistic Regression, Decision Trees, Random Forest, LightGBM, XGBoost                 
Model Evaluation Metrics: Accuracy, Precision, Recall, F1                    
Feature Engineering: One-Hot Encoding,Standard Scaling, Dimensionality Reduction                 
Deployment: Joblib, Pickle               
