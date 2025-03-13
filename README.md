# 💳 Credit Card Fraud Detection Using XGBoost  

## 🚀 **Project Overview**
This project aims to detect fraudulent credit card transactions using machine learning. We experimented with multiple models and **found that the default XGBoost model performed the best**. Despite extensive hyperparameter tuning, default XGBoost provided the best balance of **precision, recall, and F1-score** for fraud classification.

---

## 📂 **Dataset Information**
- **Dataset:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Total Transactions:** 284,807  
- **Fraud Cases:** 492 (~0.17% of total transactions)  
- **Non-Fraud Cases:** 284,315  
- **Features:**  
  - 30 anonymized features (`V1` to `V28`) derived using **PCA**  
  - `Time`: Seconds elapsed since the first transaction  
  - `Amount`: Transaction amount  
  - `Class`: **Target variable (1 = Fraud, 0 = Non-Fraud)**  

---

## 🔬 **Project Methodology**
1. **Data Preprocessing**
   - Standardized the `Amount` feature.
   - Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance fraud cases in training data.

2. **Model Training & Testing**
   - Tried different models (**Logistic Regression, Decision Tree, LightGBM, Neural Networks**).
   - **XGBoost outperformed all others**.
   - Used **GridSearchCV** to fine-tune hyperparameters.

3. **Final Model Selection**
   - Compared **Default XGBoost vs Hyperparameter-Tuned XGBoost**.
   - Surprisingly, **Default XGBoost gave better results**.
   - **Final decision:** Use **default XGBoost** with minor threshold tuning.

---

## 📊 **Model Performance Comparison**
| **Model**             | **Precision (Fraud)** | **Recall (Fraud)** | **F1-Score (Fraud)** |
|---------------------|--------------------|-----------------|-----------------|
| **Default XGBoost** | **0.79**           | **0.85**        | **0.82**        |
| **Tuned XGBoost**   | **0.35**           | **0.88**        | **0.50**        |

✅ **Default XGBoost was superior** because:  
- It had a **higher F1-score (0.75 vs 0.50)**.  
- It maintained a **strong fraud recall (0.86)** without excessive false positives.  
- The tuned model had **very low precision (0.35)**, leading to **too many false alarms**.

---

## 🚀 **Final Model: Default XGBoost**
### **🔹 Why Did Hyperparameter Tuning Fail?**
- **`scale_pos_weight=7` hurt precision** by prioritizing fraud cases too much.  
- **`max_depth=5 & n_estimators=150` overfitted the training data**.  
- **Default settings provided a well-balanced model**, making extra tuning unnecessary.

## 📈 **Final ROC-AUC & Precision-Recall Curve**
After evaluating the model, we analyzed the **ROC-AUC Curve** and **Precision-Recall Curve** to measure the fraud detection performance.

### **🔹 ROC-AUC Score (Receiver Operating Characteristic - Area Under Curve)**
- **ROC-AUC Score:** **0.97**
- A high **ROC-AUC score indicates strong discrimination** between fraud and non-fraud cases.
- The model effectively **minimizes false positives while capturing fraudulent transactions**.

### **🔹 Precision-Recall Curve Analysis**
- **Average Precision (AP) Score:** **0.77**
- This curve is especially useful for highly **imbalanced datasets** like fraud detection.
- The model maintains **high precision at lower recall values**, ensuring fraud is detected efficiently.

| **Metric**  | **Value** |
|------------|----------|
| **ROC-AUC** | 0.99 |
| **PR AUC**  | 0.85 |

✅ **These results confirm that the model successfully detects fraud with minimal false positives.**

---

## 🎯 **Key Takeaways**
1️⃣ **Default XGBoost performed better than hyperparameter-tuned versions.**  
2️⃣ **High recall (86%) ensures most fraud cases are detected.**  
3️⃣ **Precision (67%) is reasonable, balancing fraud detection and false alarms.**  
4️⃣ **Over-tuning `scale_pos_weight` and deep trees led to overfitting.**  
5️⃣ **Sometimes, default settings work best!**  

---

## 📌 **Next Steps for Further Improvement**
🔹 **Test Ensemble Learning** (XGBoost + LightGBM + Neural Network)  
🔹 **Try a Hybrid Sampling Approach** (SMOTE + Undersampling)  
🔹 **Experiment with AutoML-based tuning for even better performance**  

---
