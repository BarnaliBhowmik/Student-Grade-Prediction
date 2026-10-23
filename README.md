
# Student Grade Prediction Using Decision Trees
<img width="1983" height="793" alt="Student Grade Prediction Banner" src="https://github.com/user-attachments/assets/3473350a-0689-477d-908e-81ba42866c7a" />

## 1. Project Overview

**Objective:** Predict student academic performance categories using machine learning classification.

- **Dataset:** UCI Student Performance — Mathematics
- **Records:** 395 students
- **Input features:** Demographic, social, school-related, and academic attributes
- **Target:** Fail, Satisfactory, Good, Excellent
- **Models:** Custom C4.5-style, Gini, and Entropy decision trees
- **Status:** Model comparison and initial tuning completed; final evaluation pending.

## 2. Problem Statement

Predict students' final mathematics performance category from available student attributes. Compare performance with and without previous grades (G1 and G2) to assess their predictive contribution.

## 3. Dataset and Preprocessing

| Property | Details |
|---|---|
| Dataset size | 395 rows, 33 columns |
| Missing values | None found |
| Target | `grade_category`, derived from G3 |
| Fail | G3 = 0–9 |
| Satisfactory | G3 = 10–13 |
| Good | G3 = 14–16 |
| Excellent | G3 = 17–20 |

**Leakage prevention:** `G3` and `grade_category` are excluded from input features. G1 and G2 are included only in the experiment that uses previous grades.

## 4. Methodology

1. Inspect and preprocess the dataset.
2. Prepare features with and without G1/G2.
3. Implement a custom C4.5-style decision tree using entropy, information gain, and gain ratio.
4. Compare it with standard Gini and Entropy decision trees.
5. Evaluate using accuracy, macro F1, classification reports, and confusion matrices.
6. Perform stratified five-fold cross-validation.
7. Tune hyperparameters using GridSearchCV.
8. Implement a separate train-validation-test evaluation workflow.

## 5. Model Comparison

### Experiment A: With Previous Grades

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Custom C4.5-style | 59.49% | 0.6358 |
| Gini | **79.75%** | **0.8174** |
| Entropy | 75.95% | 0.7588 |

### Experiment B: Without Previous Grades

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Custom C4.5-style | **40.51%** | 0.2197 |
| Gini | 39.24% | **0.3791** |
| Entropy | 36.71% | 0.3120 |

**Finding:** Prediction performance decreased substantially without G1 and G2.

## 6. Five-Fold Cross-Validation

| Model | Mean Accuracy | Mean Macro F1 |
|---|---:|---:|
| Custom C4.5-style — with G1/G2 | 0.5544 | 0.5733 |
| Gini — with G1/G2 | **0.8152 ± 0.0405** | **0.8347** |
| Entropy — with G1/G2 | 0.8076 ± 0.0449 | 0.8134 |
| Custom C4.5-style — without G1/G2 | 0.4278 | 0.2341 |
| Gini — without G1/G2 | 0.4101 ± 0.0062 | **0.3084** |
| Entropy — without G1/G2 | 0.3772 ± 0.0539 | 0.3003 |

**Finding:** The standard Gini tree achieved the highest mean macro F1 in both experiments.

## 7. Hyperparameter Tuning

GridSearchCV evaluated 80 parameter configurations using five-fold stratified cross-validation, optimizing macro F1.

| Parameter | Selected Value |
|---|---|
| Criterion | Gini |
| Maximum depth | 5 |
| Minimum samples per leaf | 6 |
| Minimum samples to split | 2 |
| Mean CV accuracy | 0.8582 |
| Mean CV macro F1 | **0.8665** |
| Macro F1 standard deviation | 0.0456 |

These are cross-validation results, not independent test results.

## 8. Baseline vs Tuned Model

Results from the initial 80/20 train-test evaluation:

| Metric | Baseline Gini | Tuned Gini |
|---|---:|---:|
| Accuracy | 79.75% | **84.81%** |
| Macro F1 | 0.8174 | **0.8614** |
| Correct predictions | 63/79 | **67/79** |

The tuned model improved accuracy by **5.06 percentage points** and macro F1 by **0.0440**.

### Tuned Model: Class-Wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Fail | 1.00 | 0.58 | 0.73 |
| Satisfactory | 0.75 | 1.00 | 0.86 |
| Good | 0.94 | 1.00 | 0.97 |
| Excellent | 1.00 | 0.80 | 0.89 |

**Observation:** Good-class F1 was 0.97, while Fail recall was 0.58. The model missed some students in the Fail category. Excellent-class metrics are based on only five test examples.

**Evaluation limitation:** Hyperparameter tuning occurred before the final test split was established. Therefore, the reported 84.81% accuracy is provisional and not a fully independent estimate of generalization performance.

## 9. Tech Stack

- Python
- Pandas and NumPy
- scikit-learn
- Matplotlib
- Git and GitHub
- Flask (planned)

## 10. Project Structure

```text
Student-Grade-Prediction/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── prepare_features.py
│   ├── c45_tree.py
│   ├── train_model.py
│   ├── compare_models.py
│   ├── cross_validate.py
│   ├── evaluate_model.py
│   ├── tune_model.py
│   ├── final_evaluation.py
│   └── plot_results.py
├── static/
├── templates/
├── tests/
└── README.md
```

## 11. Future Work

- [ ] Implement stratified 80–10–10 train-validation-test splitting.
- [ ] Tune hyperparameters using training data and validation results only.
- [ ] Evaluate the selected model on the untouched test set.
- [ ] Regenerate final metrics and confusion matrix.
- [ ] Add automated tests.
- [ ] Build a Flask prediction interface.
- [ ] Update results after final evaluation.

## 12. Conclusion

The best observed result so far is **84.81% accuracy and 0.8614 macro F1** from the tuned Gini tree. Five-fold cross-validation also favored the standard Gini implementation over the custom C4.5-style tree.

Previous academic grades substantially improved prediction performance. The next priority is a leakage-conscious train-validation-test evaluation before reporting final model performance.

---

*Results reflect experiments completed so far. Final metrics will be updated after the evaluation workflow is corrected.*
