# Student Counseling Prediction System

This project aims to predict whether a student may need mental health counseling using academic, behavioral, and background data. The model provides predictive insights and uses SHAP explainability to interpret the key contributing features. Designed as an enterprise-grade pipeline.


##  Project Goals

- Predict the need for mental health counseling among students
- Apply preprocessing and feature engineering steps to clean and prepare the data
- Use models like Random Forest and XGBoost
- Improve results using SMOTE for class imbalance
- Interpret predictions with SHAP explainability



---

## 🧪 Models Used

| Model         | Accuracy | F1-Score (Class 1) | Notes                     |
|---------------|----------|--------------------|---------------------------|
| Random Forest | 82%      | 0.30               | With SMOTE applied        |
| XGBoost       | 82.3%    | 0.38               | Tuned with hyperparams    |

---

## 📊 SHAP Explainability

SHAP was used to explain the XGBoost predictions.

![SHAP Summary](![image](https://github.com/user-attachments/assets/6d405375-716c-48a6-a501-72a7925e968a)
)

> Surprisingly, `school`, `failures`, and `family relationship quality` were among the top predictors — further investigated using EDA.

---

## EDA Highlights

- Strong correlation found between `failures`, `studytime`, and `need_counselling`
- Class imbalance noted (only ~15% required counseling)
- `final_grade` negatively correlated with `need_counselling`

---

### ✅ Preprocessing
- Label and One-Hot Encoding
- Log transformation for skewed variables
- Feature engineering: `avg_parent_edu`, `low_studytime`, `high_goout`

### ✅ Modeling
- SMOTE applied to handle class imbalance
- Trained using `RandomForestClassifier` and `XGBClassifier`

### ✅ Saving Artifacts
```python
joblib.dump(model, 'models/xgb_model.pkl')




