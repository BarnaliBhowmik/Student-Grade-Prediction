
# C4.5-Based Student Grade Prediction

A machine learning project that predicts student academic performance using a decision tree based on the C4.5 algorithm and the UCI Student Performance dataset.

## Overview

This project explores how student academic records and other relevant factors can be used to predict final academic performance.

The model will use the C4.5 decision tree approach, which selects splits using the gain ratio. The implementation will be developed and tested in Python before being integrated into a web application.

## Objectives

- Explore and preprocess student performance data.
- Implement C4.5-style decision tree splitting using gain ratio.
- Predict student performance categories from selected input features.
- Evaluate the model on unseen data.
- Visualize decision rules and model performance.
- Develop a web application for entering student information and viewing predictions.

## Dataset

**Source:** UCI Machine Learning Repository

**Dataset:** Student Performance

**File:** `student-mat.csv`

Official source:
https://archive.ics.uci.edu/dataset/320/student+performance

The dataset contains student demographic, social, school-related, and academic information. The `G3` attribute represents the final grade, scored from 0 to 20.

The project will document its feature selection, target-label definitions, and preprocessing decisions.

## Methodology

1. Load and inspect the dataset.
2. Clean and preprocess the data.
3. Define the prediction target and input features.
4. Implement decision tree splitting using gain ratio.
5. Train and evaluate the model.
6. Analyze decision rules and prediction errors.
7. Integrate the model into a web application.

## Evaluation

The model will be evaluated using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

A baseline model will be used for comparison. Data leakage and the effect of using earlier-period grades will also be investigated.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn for evaluation and baseline models
- Matplotlib
- Flask
- HTML, CSS, and Bootstrap
- Git and GitHub

## Planned Project Structure

```text
C45-Student-Grade-Prediction/
├── data/
├── models/
├── notebooks/
├── src/
├── templates/
├── static/
├── tests/
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Current Status

Project setup and dataset preparation.

The C4.5 implementation, model evaluation, and web application are planned development stages.

## Limitations

The dataset represents students in Portuguese schools. Predictions may not generalize to other educational systems or student populations.

Predictions are estimates, not guarantees of academic outcomes. The model is intended for educational analysis and should not be used as the sole basis for high-stakes decisions about students.

## Acknowledgements

Dataset provided by the UCI Machine Learning Repository.

Dataset reference:
Cortez, P., & Silva, A. M. G. (2008). Using Data Mining to Predict Secondary School Student Performance.
