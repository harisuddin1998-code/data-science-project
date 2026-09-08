"""
download_sample_datasets.py

This script downloads several real, well-known Kaggle datasets and copies
them into the "sample_datasets" folder inside this project.

WHY: These datasets can then be uploaded into dataset_analyzer_app.py
(the webpage we built) to show a live demo/report to your teacher, without
needing to manually search and download datasets from Kaggle yourself.

HOW: It uses the "kagglehub" library, which downloads public Kaggle
datasets straight to your computer (no Kaggle login needed for public
datasets).

To run this script:
    pip install kagglehub
    python download_sample_datasets.py
"""

import os
import shutil
import kagglehub
import pandas as pd

# -----------------------------------------------------------------
# STEP 1: The list of datasets we want to download.
# Each one is a real Kaggle dataset good for classification problems
# (i.e. predicting a Yes/No or category-type outcome) — perfect for
# our Decision Tree app.
# "kaggle_slug" is the dataset's unique Kaggle address (owner/dataset-name).
# -----------------------------------------------------------------
DATASETS = [
    {"kaggle_slug": "bhavikjikadara/mental-health-dataset",
     "save_as": "mental_health_dataset.csv"},
    {"kaggle_slug": "mlg-ulb/creditcardfraud",
     "save_as": "credit_card_fraud.csv"},
    {"kaggle_slug": "teejmahal20/airline-passenger-satisfaction",
     "save_as": "airline_passenger_satisfaction.csv"},
    {"kaggle_slug": "uciml/adult-census-income",
     "save_as": "adult_census_income.csv"},
    {"kaggle_slug": "blastchar/telco-customer-churn",
     "save_as": "telco_customer_churn.csv"},
    {"kaggle_slug": "fedesoriano/stroke-prediction-dataset",
     "save_as": "stroke_prediction.csv"},
    {"kaggle_slug": "johnsmith88/heart-disease-dataset",
     "save_as": "heart_disease.csv"},
    {"kaggle_slug": "uciml/pima-indians-diabetes-database",
     "save_as": "diabetes.csv"},
    {"kaggle_slug": "pavansubhasht/ibm-hr-analytics-attrition-dataset",
     "save_as": "employee_attrition.csv"},
    {"kaggle_slug": "uciml/mushroom-classification",
     "save_as": "mushroom_classification.csv"},
    {"kaggle_slug": "yasserh/titanic-dataset",
     "save_as": "titanic.csv"},
    {"kaggle_slug": "prakharrathi25/banking-dataset-marketing-targets",
     "save_as": "bank_marketing.csv"},
]

# Where we will save the final, ready-to-upload CSV files
OUTPUT_FOLDER = "sample_datasets"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def find_first_csv(folder_path):
    """Kaggle datasets can contain more than one file. This function
    looks inside the downloaded folder and returns the first .csv file
    it finds (the main data file)."""
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                return os.path.join(root, file_name)
    return None


# -----------------------------------------------------------------
# STEP 2: Download each dataset one by one
# -----------------------------------------------------------------
summary_rows = []

for dataset in DATASETS:
    slug = dataset["kaggle_slug"]
    save_as = dataset["save_as"]
    print(f"\nDownloading: {slug} ...")

    try:
        # kagglehub downloads the dataset and returns the folder path
        # where the files were saved on this computer
        downloaded_folder = kagglehub.dataset_download(slug)

        # Find the main CSV file inside that folder
        csv_path = find_first_csv(downloaded_folder)
        if csv_path is None:
            print(f"  Skipped: no CSV file found for {slug}")
            continue

        # Copy it into our project's sample_datasets folder with a clean name
        destination_path = os.path.join(OUTPUT_FOLDER, save_as)
        shutil.copyfile(csv_path, destination_path)

        # Quickly check how many rows/columns it has, just for our summary
        df = pd.read_csv(destination_path, nrows=None)
        summary_rows.append({
            "file": save_as,
            "kaggle_source": slug,
            "rows": len(df),
            "columns": len(df.columns),
        })
        print(f"  Saved: {destination_path}  ({len(df)} rows, {len(df.columns)} columns)")

    except Exception as error:
        # If a dataset fails to download (e.g. removed, renamed, needs
        # login), we just skip it and continue with the rest
        print(f"  Skipped: could not download {slug} ({error})")

# -----------------------------------------------------------------
# STEP 3: Print a final summary table of everything that was downloaded
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_FOLDER, "_summary.csv"), index=False)
print(f"\nAll files saved inside the '{OUTPUT_FOLDER}' folder.")
print("Upload any of them into dataset_analyzer_app.py to see the report.")
