# Task 4: Logistic Regression - Breast Cancer Classification

## 📌 Objective

To implement a **binary classification** model using **logistic regression** to classify breast cancer as **malignant** or **benign** using the **Breast Cancer Wisconsin Dataset**.

## 📂 Dataset

- [Breast Cancer Wisconsin Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 🛠 Tools & Libraries Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 🔍 Project Steps

1. Loaded the dataset and preprocessed it (removed ID columns, encoded labels).
2. Split into train and test sets (80/20).
3. Standardized the features using `StandardScaler`.
4. Trained a **Logistic Regression** model.
5. Evaluated using:
   - Confusion Matrix
   - Precision & Recall
   - ROC-AUC Score
   - Classification Report
6. Plotted ROC Curve to visualize model performance.

## 📈 Evaluation

- **Precision**: ~0.97  
- **Recall**: ~0.95  
- **ROC-AUC Score**: ~0.99  
- High performance shows the model can distinguish between malignant and benign tumors effectively.

## 📊 Visualization

- **ROC Curve** plotted showing high area under the curve indicating good classification power.
- ![image](https://github.com/user-attachments/assets/1dff964d-6434-404e-9e22-2e432392a72b)


# Interview Questions - Logistic Regression

### 1. How does logistic regression differ from linear regression?
- Logistic regression is used for **classification**, not regression.
- Output of linear regression is continuous; logistic regression outputs probabilities (0 to 1).

---

### 2. What is the sigmoid function?
- A function that maps any real number to a value between 0 and 1.
- Formula: `1 / (1 + e^(-z))`
- Used to convert model output to probability.

---

### 3. What is precision vs recall?
- **Precision**: TP / (TP + FP) – How many predicted positives are truly positive.
- **Recall**: TP / (TP + FN) – How many actual positives are predicted correctly.

---

### 4. What is the ROC-AUC curve?
- ROC (Receiver Operating Characteristic) curve plots TPR vs FPR.
- AUC (Area Under Curve) quantifies the overall performance of the model (closer to 1 is better).

---

### 5. What is the confusion matrix?
A 2x2 table that summarizes:
|            | Predicted Positive | Predicted Negative |
|------------|--------------------|--------------------|
| Actual Pos | True Positives (TP) | False Negatives (FN) |
| Actual Neg | False Positives (FP)| True Negatives (TN) |

---

### 6. What happens if classes are imbalanced?
- Model may become biased toward the majority class.
- Use techniques like:
  - Oversampling / Undersampling
  - Class weights
  - SMOTE

---

### 7. How do you choose the threshold?
- Default is 0.5; change it based on precision-recall trade-off.
- Use ROC curve or Precision-Recall curve to pick an optimal threshold.

---

### 8. Can logistic regression be used for multi-class problems?
Yes, using:
- One-vs-Rest (OvR)
- Multinomial logistic regression
