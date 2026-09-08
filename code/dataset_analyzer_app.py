"""
dataset_analyzer_app.py

A SIMPLE webpage (Streamlit) where you can:
1. Upload ANY dataset (CSV file) - for example, one downloaded from GitHub/Kaggle.
2. Pick which column is the "target" (the thing you want to predict).
3. Pick which columns are "features" (the inputs used to predict the target).
4. The app trains a Decision Tree Classifier on that data.
5. The app shows you a full report: test accuracy, a confusion matrix graph,
   a table comparing actual vs predicted values on the test set (X_test/y_test),
   and a picture of the actual decision tree the model built.

To run this app:
    streamlit run dataset_analyzer_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------
st.title("Dataset Analyzer - Decision Tree Classifier")
st.write(
    "Upload a CSV dataset, choose the column you want to predict, and this "
    "app will train a Decision Tree model on it and show you a full report."
)

# -----------------------------------------------------------------
# STEP 1: Upload the dataset
# -----------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your dataset (.csv file)", type=["csv"])

# Everything below only runs once a file has been uploaded
if uploaded_file is not None:

    # Read the uploaded CSV into a table (DataFrame)
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Preview of Your Dataset")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head(10))

    # Drop rows that have missing values, to keep things simple
    df = df.dropna()

    # -----------------------------------------------------------------
    # STEP 2: Let the user choose the target column and feature columns
    # -----------------------------------------------------------------
    st.subheader("2. Choose Target and Feature Columns")

    all_columns = list(df.columns)
    target_column = st.selectbox(
        "Which column do you want to PREDICT (target)?",
        options=all_columns,
        index=len(all_columns) - 1,  # default: last column
    )

    feature_options = [col for col in all_columns if col != target_column]
    feature_columns = st.multiselect(
        "Which columns should be used as FEATURES (inputs)?",
        options=feature_options,
        default=feature_options,
    )

    max_depth = st.slider(
        "Decision Tree depth (smaller = simpler tree, larger = more accurate but more complex)",
        min_value=2, max_value=10, value=4,
    )

    train_button = st.button("Train Decision Tree Model")

    if train_button and len(feature_columns) > 0:

        # -----------------------------------------------------------------
        # STEP 3: Prepare the data (X = features, y = target)
        # -----------------------------------------------------------------
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Machine learning models only understand numbers. If a feature column
        # contains text (e.g. "Male"/"Female", "Poor"/"Good"), convert it to
        # numbers using LabelEncoder. We check "is this column NOT numeric"
        # instead of checking for a specific text dtype name, because
        # different pandas versions label text columns differently
        # (e.g. "object" or "str") - this way it catches text columns either way.
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # If the target column is text (e.g. "Yes"/"No", "Low"/"Medium"/"High"),
        # convert it to numbers too, and remember the original labels so we can
        # show readable names later.
        target_encoder = None
        if not pd.api.types.is_numeric_dtype(y):
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y.astype(str))
            class_names = [str(c) for c in target_encoder.classes_]
        else:
            class_names = [str(c) for c in sorted(y.unique())]

        # -----------------------------------------------------------------
        # STEP 4: Split into training data and testing data
        # -----------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -----------------------------------------------------------------
        # STEP 5: Train the Decision Tree model
        # -----------------------------------------------------------------
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # -----------------------------------------------------------------
        # STEP 6: Test the model and build the report
        # -----------------------------------------------------------------
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)

        st.subheader("3. Model Report")
        st.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")

        # Not every class is guaranteed to show up in the test set (this can
        # happen with rare categories or small datasets). classification_report
        # and the confusion matrix need their label list to match exactly what
        # actually appears in y_test/y_pred, otherwise they crash - so we build
        # that list from the real data instead of assuming every class is there.
        labels_present = sorted(set(y_test) | set(y_pred))
        names_present = [class_names[i] for i in labels_present]

        st.write("**Classification Report** (precision, recall, f1-score per class):")
        report_dict = classification_report(
            y_test, y_pred, labels=labels_present, target_names=names_present, output_dict=True
        )
        st.dataframe(pd.DataFrame(report_dict).transpose())

        # ----- Confusion Matrix graph -----
        st.write("**Confusion Matrix** (rows = actual, columns = predicted):")
        cm = confusion_matrix(y_test, y_pred, labels=labels_present)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names_present, yticklabels=names_present, ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # ----- X_test / y_test / predictions table -----
        st.write("**Test Set: Actual vs Predicted** (first 15 rows):")
        results_table = X_test.copy()
        results_table["Actual"] = [class_names[i] for i in y_test]
        results_table["Predicted"] = [class_names[i] for i in y_pred]
        st.dataframe(results_table.head(15))

        # ----- Decision Tree structure picture -----
        st.subheader("4. Decision Tree Structure")
        st.write("This is exactly how the model makes its decisions, step by step:")
        # Same idea here: the tree only knows about classes it saw during
        # training, so we label it using model.classes_ (not the full
        # class_names list) to keep the labels lined up correctly.
        tree_class_names = [class_names[i] for i in model.classes_]
        fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
        plot_tree(
            model,
            feature_names=feature_columns,
            class_names=tree_class_names,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax_tree,
        )
        st.pyplot(fig_tree)

    elif train_button and len(feature_columns) == 0:
        st.error("Please select at least one feature column before training.")

else:
    st.info("Upload a CSV file above to get started.")
