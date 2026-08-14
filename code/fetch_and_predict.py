"""
fetch_and_predict.py

This is the "backend" of our project.

FLOW OF THE WHOLE PROJECT (explain this in viva):
1. User fills a GOOGLE FORM (the input form) with their daily routine.
2. Google Form automatically saves every response as a new row in a
   linked GOOGLE SHEET.
3. That Google Sheet is "Published to the web" as a CSV link (a plain
   text file that updates automatically whenever a new response comes in).
4. THIS SCRIPT reads that CSV link, looks at the newest response,
   passes it into our already-trained Machine Learning model
   (stress_model.pkl), and prints the predicted Stress Level.

No website/app was coded by us for the input form — Google Form does
that part for us. We only wrote the small "brain" that reads the data
and makes the prediction. This keeps the project simple and easy to
explain.
"""

import time
import pandas as pd
import joblib

# -----------------------------------------------------------------
# STEP 0: SETTINGS - change this one line after you create your form
# -----------------------------------------------------------------
# How to get this link:
#   1. Open your Google Form -> Responses tab -> click the green Sheets icon
#      to create a linked Google Sheet.
#   2. Open that Google Sheet -> File -> Share -> Publish to web.
#   3. Under "Link", choose the responses sheet, choose "Comma-separated
#      values (.csv)", then click Publish.
#   4. Copy the link it gives you and paste it below.
SHEET_CSV_URL = "PASTE_YOUR_PUBLISHED_GOOGLE_SHEET_CSV_LINK_HERE"

# How often (in seconds) the script checks the sheet for a new response.
CHECK_EVERY_SECONDS = 15

# This must match the EXACT question text you used in your Google Form,
# because Google Sheets names each column after the question text.
COLUMN_SLEEP = "How many hours do you sleep per day?"
COLUMN_STUDY_WORK = "How many hours do you study or work per day?"
COLUMN_SCREEN = "How many hours do you spend on screens per day?"
COLUMN_ACTIVITY = "How many hours of physical activity do you get per week?"
COLUMN_PRESSURE = "Rate your academic/work pressure (1 = very low, 10 = very high)"
COLUMN_DIET = "How would you rate your diet quality?"

# A small local file used to remember how many responses we already
# processed, so we don't predict the same response twice.
PROGRESS_FILE = "last_processed_count.txt"

# -----------------------------------------------------------------
# STEP 1: Load our already-trained model (created earlier by train_model.py)
# -----------------------------------------------------------------
model = joblib.load("stress_model.pkl")
scaler = joblib.load("scaler.pkl")
diet_encoder = joblib.load("diet_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")


def get_last_processed_count():
    """Read how many responses we already handled before."""
    try:
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 0


def save_last_processed_count(count):
    """Remember how many responses we have handled so far."""
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(count))


def predict_stress(row):
    """
    Takes ONE row (one form response) and returns the predicted
    stress level as text: "Low", "Medium", or "High".
    """
    # Convert the diet quality text ("Poor"/"Average"/"Good") into the
    # same numeric code used during training
    diet_encoded = diet_encoder.transform([row[COLUMN_DIET]])[0]

    # Arrange values in the exact same column order used in training
    input_data = pd.DataFrame([{
        "Sleep_Hours": row[COLUMN_SLEEP],
        "Study_Work_Hours": row[COLUMN_STUDY_WORK],
        "Screen_Time": row[COLUMN_SCREEN],
        "Physical_Activity": row[COLUMN_ACTIVITY],
        "Pressure_Score": row[COLUMN_PRESSURE],
        "Diet_Quality_Encoded": diet_encoded,
    }])

    # Scale the input the same way the training data was scaled
    input_scaled = scaler.transform(input_data)

    # Predict (returns an encoded number, e.g. 0, 1, or 2)
    prediction_encoded = model.predict(input_scaled)[0]

    # Convert the number back to a readable label
    prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]
    return prediction_label


def check_for_new_responses():
    """
    Downloads the latest data from the published Google Sheet CSV link,
    and predicts the stress level for any NEW response(s) since last time.
    """
    df = pd.read_csv(SHEET_CSV_URL)

    already_processed = get_last_processed_count()
    total_rows = len(df)

    if total_rows <= already_processed:
        print("No new form responses yet...")
        return

    # Only look at the rows we haven't processed yet
    new_rows = df.iloc[already_processed:total_rows]

    for _, row in new_rows.iterrows():
        prediction = predict_stress(row)
        print("-" * 50)
        print("New form response received!")
        print(f"Sleep Hours        : {row[COLUMN_SLEEP]}")
        print(f"Study/Work Hours   : {row[COLUMN_STUDY_WORK]}")
        print(f"Screen Time        : {row[COLUMN_SCREEN]}")
        print(f"Physical Activity  : {row[COLUMN_ACTIVITY]}")
        print(f"Pressure Score     : {row[COLUMN_PRESSURE]}")
        print(f"Diet Quality       : {row[COLUMN_DIET]}")
        print(f"PREDICTED STRESS LEVEL: {prediction}")
        print("-" * 50)

    # Update our progress so we don't repeat these rows next time
    save_last_processed_count(total_rows)


# -----------------------------------------------------------------
# MAIN LOOP: keep checking the Google Sheet for new responses
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("Stress Predictor backend started.")
    print(f"Watching Google Sheet for new form responses every {CHECK_EVERY_SECONDS} seconds...")
    print("Press CTRL+C to stop.\n")

    while True:
        check_for_new_responses()
        time.sleep(CHECK_EVERY_SECONDS)
