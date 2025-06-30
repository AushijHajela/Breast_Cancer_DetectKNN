***Breast Cancer Detection using K-Nearest Neighbors (KNN)***
***Overview***
This project applies machine learning techniques to predict whether a tumor is benign (B) or malignant (M) using the Breast Cancer Wisconsin dataset. The model uses K-Nearest Neighbors (KNN) for classification and achieves high accuracy and AUC.

***Files***
detect.ipynb: Main Jupyter notebook with data analysis, model training, evaluation, and tuning.

breast-cancer.csv: Dataset used for training/testing the model.


***Key Features***
Exploratory data analysis (EDA)

Data preprocessing (label encoding, feature scaling)

Model training using KNN

GridSearchCV for hyperparameter tuning

Evaluation with:

Confusion Matrix

Classification Report

ROC AUC Curve

Threshold tuning for better recall on malignant cases

***Model Performance***
Metric	Value
Accuracy	95.1%
Precision (Malignant)	0.98
Recall (Malignant)	0.89
F1-Score (Malignant)	0.93
ROC AUC Score	0.991

***High AUC and accuracy indicate that the model can effectively distinguish between benign and malignant tumors.***

***Libraries Used***
pandas
numpy
matplotlib
seaborn
scikit-learn

***Model Comparison***
Model	Accuracy
Logistic Regression	97.20%
Support Vector Machine (SVM)	96.50%
K-Nearest Neighbors (KNN)	95.10%
Random Forest	95.10%

***Logistic Regression gave the highest accuracy, but all models performed well and were evaluated using additional metrics like recall, precision, and AUC.***
