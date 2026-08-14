"""
generate_sample_data.py

This script creates a SAMPLE dataset (fake but realistic-looking data)
so you can run the whole project immediately, even before you download
the real dataset from Kaggle.

Once you download the real Kaggle dataset, just replace the file
'data/stress_dataset.csv' with the real one (using the SAME column names
used below), and everything else (train_model.py, app.py) will still work.
"""

import numpy as np
import pandas as pd
import os

# Set a "seed" so that random numbers are the same every time we run this
# (this makes our results reproducible / repeatable)
np.random.seed(42)

# How many fake student/professional records we want to create
NUM_ROWS = 500

# Step 1: Create random values for each lifestyle feature
sleep_hours = np.random.uniform(3, 10, NUM_ROWS)          # 3 to 10 hours of sleep
study_work_hours = np.random.uniform(1, 12, NUM_ROWS)     # 1 to 12 hours of study/work
screen_time = np.random.uniform(1, 14, NUM_ROWS)          # 1 to 14 hours of screen time
physical_activity = np.random.uniform(0, 7, NUM_ROWS)     # 0 to 7 hours of exercise per week
pressure_score = np.random.randint(1, 11, NUM_ROWS)       # pressure score from 1 to 10
diet_quality = np.random.choice(["Poor", "Average", "Good"], NUM_ROWS)

# Step 2: Turn diet quality into a number so we can use it in a formula
# Poor = 0, Average = 1, Good = 2
diet_quality_score = pd.Series(diet_quality).map({"Poor": 0, "Average": 1, "Good": 2}).values

# Step 3: Create a simple "stress score" using a basic weighted formula.
# This is just a rule we made up so the fake data makes logical sense:
# - less sleep -> more stress
# - more study/work hours -> more stress
# - more screen time -> more stress
# - more physical activity -> less stress
# - higher pressure score -> more stress
# - better diet -> less stress
stress_score = (
    (10 - sleep_hours) * 1.5
    + study_work_hours * 1.2
    + screen_time * 1.0
    - physical_activity * 1.3
    + pressure_score * 2.0
    - diet_quality_score * 2.0
)

# add a little random noise so it's not a "perfect" formula (more realistic)
stress_score = stress_score + np.random.normal(0, 5, NUM_ROWS)

# Step 4: Convert the numeric stress_score into 3 categories: Low, Medium, High
# using simple cut-off points (percentiles)
low_cutoff = np.percentile(stress_score, 33)
high_cutoff = np.percentile(stress_score, 66)

def score_to_label(score):
    if score <= low_cutoff:
        return "Low"
    elif score <= high_cutoff:
        return "Medium"
    else:
        return "High"

stress_level = [score_to_label(s) for s in stress_score]

# Step 5: Put everything into a single table (DataFrame)
df = pd.DataFrame({
    "Sleep_Hours": sleep_hours.round(1),
    "Study_Work_Hours": study_work_hours.round(1),
    "Screen_Time": screen_time.round(1),
    "Physical_Activity": physical_activity.round(1),
    "Pressure_Score": pressure_score,
    "Diet_Quality": diet_quality,
    "Stress_Level": stress_level,
})

# Step 6: Save this table as a CSV file inside a "data" folder
os.makedirs("data", exist_ok=True)
df.to_csv("data/stress_dataset.csv", index=False)

print("Sample dataset created successfully at: data/stress_dataset.csv")
print(df.head())
print("\nClass balance:\n", df["Stress_Level"].value_counts())
