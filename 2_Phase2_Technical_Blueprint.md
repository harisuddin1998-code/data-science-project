# Phase 2 – Technical Blueprint
## Mental Health & Stress Level Predictor — Data Science Pipeline

---

## 1. Data Preparation & Preprocessing

1. **Load the dataset** using `pandas.read_csv()`.
2. **Inspect the data:** check shape, column names, data types (`df.info()`), and first few rows (`df.head()`).
3. **Handle missing values:**
   - Numeric columns → fill with mean/median (`df['col'].fillna(df['col'].mean())`).
   - Categorical columns → fill with mode, or drop rows if very few are missing.
4. **Handle duplicates:** remove duplicate rows (`df.drop_duplicates()`).
5. **Encode categorical variables:**
   - Ordinal features (e.g., Diet Quality: Poor/Average/Good) → `LabelEncoder` or manual mapping.
   - Nominal features (if any, e.g., Gender) → `OneHotEncoder` / `pd.get_dummies()`.
6. **Encode the target variable:** `Stress_Level` (Low/Medium/High) → 0/1/2 using `LabelEncoder`.
7. **Feature scaling:** apply `StandardScaler` or `MinMaxScaler` to numeric columns (important for Logistic Regression; less critical for Random Forest but doesn't hurt).
8. **Train-test split:** typically 80% train / 20% test using `train_test_split(..., stratify=y)` to preserve class balance.

---

## 2. Exploratory Data Analysis (EDA)

**Summary statistics:**
- `df.describe()` for numeric spread (mean, std, min/max) of sleep hours, study hours, screen time, etc.
- `df['Stress_Level'].value_counts()` to check class balance (Low/Medium/High).

**Suggested graphs:**
| Graph | Purpose |
|---|---|
| Count plot of `Stress_Level` | Check class imbalance |
| Histogram/KDE of Sleep Hours, Screen Time, Study Hours | Understand distribution |
| Box plots of each feature grouped by Stress Level | See how each factor differs across stress classes |
| Correlation heatmap (`seaborn.heatmap`) | Identify which numeric features correlate most with stress |
| Scatter plot: Sleep Hours vs Stress Level (colored) | Visualize relationship |
| Bar chart: Average Screen Time per Stress category | Compare lifestyle habits across groups |

**Key questions EDA should answer:**
- Which factor shows the strongest visual separation between Low vs High stress (e.g., do High-stress individuals sleep noticeably less)?
- Is the dataset balanced across the three stress classes, or does it need balancing (e.g., `class_weight='balanced'`, oversampling)?
- Are there outliers (e.g., someone reporting 20 study hours/day) that need capping or removal?

---

## 3. Feature Selection

Based on domain relevance and correlation analysis, the following features are proposed as model inputs:

| Feature | Type | Expected Relationship with Stress |
|---|---|---|
| Sleep Hours | Numeric | Lower sleep → higher stress |
| Study/Work Hours | Numeric | Higher hours → higher stress |
| Screen Time | Numeric | Higher screen time → higher stress |
| Physical Activity (mins/week or frequency) | Numeric/Categorical | More activity → lower stress |
| Academic/Work Pressure Score (1–10) | Numeric | Higher score → higher stress |
| Diet Quality | Categorical (encoded) | Poor diet → higher stress |

Optional: use `SelectKBest`, feature importance from `RandomForestClassifier`, or a correlation matrix to confirm/refine which features matter most before finalizing the model input set.

---

## 4. Machine Learning Classification Models

| Model | Why Consider It |
|---|---|
| **Logistic Regression** | Simple, interpretable baseline for multi-class classification (`multi_class='ovr'` or `'multinomial'`) |
| **Decision Tree Classifier** | Easy to visualize and explain decision rules to a non-technical audience |
| **Random Forest Classifier** | Handles non-linear relationships well, robust to outliers, usually higher accuracy — good "main model" for this project |
| **K-Nearest Neighbors (optional)** | Simple distance-based baseline for comparison |

**Evaluation metrics:**
- Accuracy, Precision, Recall, F1-score (use `classification_report`)
- Confusion Matrix (`ConfusionMatrixDisplay`) — especially important since this is a 3-class problem
- Cross-validation (`cross_val_score`, k=5) for more reliable performance estimate

**Recommended final choice:** Random Forest Classifier — good balance of accuracy and robustness without heavy tuning, while Logistic Regression is kept as the interpretable baseline for comparison in the report.

---

## 5. Deployment / Interface

- Save trained model using `joblib.dump(model, 'stress_model.pkl')`.
- **Input layer:** a **Google Form** collects the user's routine (sleep hours, study/work hours, screen time, physical activity, pressure score, diet quality) — no custom UI code needed.
- **Storage layer:** Google Form responses flow automatically into a linked **Google Sheet**, which is published to the web as a live-updating CSV link.
- **Backend layer:** a Python script (`fetch_and_predict.py`) periodically reads the published CSV, loads the saved model + scaler/encoders, transforms the newest response, calls `model.predict()`, and prints the predicted Stress Level (Low/Medium/High) for that response.
- *(Optional alternative interface: a basic Streamlit form, `app.py`, for local demo without Google Forms.)*
