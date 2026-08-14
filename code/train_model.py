"""
train_model.py

This is the MAIN training script for our Mental Health & Stress Level Predictor.

What this script does, step by step:
1. Load the dataset (data/stress_dataset.csv)
2. Clean and prepare the data (handle missing values, encode text columns)
3. Explore the data a little (basic EDA prints + a couple of graphs)
4. Split data into training set and testing set
5. Train two models: Logistic Regression and Random Forest
6. Evaluate both models and print accuracy / classification report
7. Save the BEST model + the encoders/scaler to disk, so the Streamlit
   app (app.py) can load them later and make predictions on new user input.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------------------
# STEP 1: Load the dataset
# -----------------------------------------------------------------
df = pd.read_csv("data/stress_dataset.csv")
print("Dataset loaded. Shape:", df.shape)
print(df.head())

# -----------------------------------------------------------------
# STEP 2: Basic cleaning
# -----------------------------------------------------------------
# Check for missing values (in a real Kaggle dataset there might be some)
print("\nMissing values per column:\n", df.isnull().sum())

# If there were missing numeric values, we would fill them with the mean, e.g.:
# df['Sleep_Hours'] = df['Sleep_Hours'].fillna(df['Sleep_Hours'].mean())
# (Not needed here since our sample data has no missing values)

# Remove exact duplicate rows, if any
df = df.drop_duplicates()

# -----------------------------------------------------------------
# STEP 3: Basic EDA (Exploratory Data Analysis)
# -----------------------------------------------------------------
print("\nStress level counts:\n", df["Stress_Level"].value_counts())

# A simple count plot of how many students fall into each stress category
plt.figure(figsize=(6, 4))
sns.countplot(x="Stress_Level", data=df, order=["Low", "Medium", "High"])
plt.title("Number of People per Stress Level")
plt.savefig("eda_stress_level_counts.png")
plt.close()

# A correlation heatmap of the numeric columns (helps us see which
# features are most related to each other)
plt.figure(figsize=(6, 5))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png")
plt.close()

print("\nEDA graphs saved as PNG files (eda_stress_level_counts.png, eda_correlation_heatmap.png)")

# -----------------------------------------------------------------
# STEP 4: Encode categorical columns into numbers
# -----------------------------------------------------------------
# Machine learning models only understand numbers, not text like "Poor"/"Good"
# or "Low"/"Medium"/"High". So we convert them using LabelEncoder.

diet_encoder = LabelEncoder()
df["Diet_Quality_Encoded"] = diet_encoder.fit_transform(df["Diet_Quality"])
# Example mapping created automatically: Average=0, Good=1, Poor=2 (alphabetical order)

target_encoder = LabelEncoder()
df["Stress_Level_Encoded"] = target_encoder.fit_transform(df["Stress_Level"])
# Example mapping: High=0, Low=1, Medium=2 (alphabetical order)

print("\nDiet Quality classes:", list(diet_encoder.classes_))
print("Stress Level classes:", list(target_encoder.classes_))

# -----------------------------------------------------------------
# STEP 5: Select our input features (X) and target/output (y)
# -----------------------------------------------------------------
feature_columns = [
    "Sleep_Hours",
    "Study_Work_Hours",
    "Screen_Time",
    "Physical_Activity",
    "Pressure_Score",
    "Diet_Quality_Encoded",
]

X = df[feature_columns]
y = df["Stress_Level_Encoded"]

# -----------------------------------------------------------------
# STEP 6: Scale numeric features
# -----------------------------------------------------------------
# Scaling makes all numbers roughly the same range (helps Logistic Regression
# perform better). Random Forest doesn't strictly need this, but it doesn't hurt.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------
# STEP 7: Split into training set and testing set
# -----------------------------------------------------------------
# 80% of the data is used to TRAIN the model, 20% is kept aside to TEST
# how well the model performs on data it has never seen before.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining rows: {len(X_train)}, Testing rows: {len(X_test)}")

# -----------------------------------------------------------------
# STEP 8: Train Model 1 - Logistic Regression (simple baseline model)
# -----------------------------------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
log_predictions = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_predictions)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", round(log_accuracy, 3))
print(classification_report(y_test, log_predictions, target_names=target_encoder.classes_))

# -----------------------------------------------------------------
# STEP 9: Train Model 2 - Random Forest (usually more accurate)
# -----------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)

print("\n--- Random Forest Results ---")
print("Accuracy:", round(rf_accuracy, 3))
print(classification_report(y_test, rf_predictions, target_names=target_encoder.classes_))

# Confusion matrix for the Random Forest model (shows correct vs incorrect predictions)
cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("eda_confusion_matrix.png")
plt.close()

# -----------------------------------------------------------------
# STEP 10: Pick the best model and save everything needed for prediction
# -----------------------------------------------------------------
if rf_accuracy >= log_accuracy:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = log_model
    best_model_name = "Logistic Regression"

print(f"\nBest model selected: {best_model_name}")

# We save 4 things to disk so the Streamlit app can reuse them later:
# 1. The trained model itself
# 2. The scaler (so new user input gets scaled the same way as training data)
# 3. The diet quality encoder (to convert "Poor"/"Average"/"Good" text into numbers)
# 4. The target encoder (to convert predicted number back into "Low"/"Medium"/"High")
joblib.dump(best_model, "stress_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(diet_encoder, "diet_encoder.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("\nModel and preprocessing objects saved successfully!")
print("Files created: stress_model.pkl, scaler.pkl, diet_encoder.pkl, target_encoder.pkl")
