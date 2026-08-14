# Setting Up the Google Form (Input Side)

This project no longer needs a custom web app for input — we use a
**Google Form** instead. This is simpler, looks realistic for a
student project, and is easy to explain in viva.

## Step 1: Create the Google Form

Go to [forms.google.com](https://forms.google.com) and create a new
form titled **"Mental Health & Stress Level Predictor"**. Add these
6 questions **exactly as written below** (the wording must match,
because the backend script matches columns by question text):

| # | Question (Question Title) | Answer Type | Notes |
|---|---|---|---|
| 1 | How many hours do you sleep per day? | Short answer (Number validation) | e.g. 0–12 |
| 2 | How many hours do you study or work per day? | Short answer (Number validation) | e.g. 0–16 |
| 3 | How many hours do you spend on screens per day? | Short answer (Number validation) | e.g. 0–16 |
| 4 | How many hours of physical activity do you get per week? | Short answer (Number validation) | e.g. 0–14 |
| 5 | Rate your academic/work pressure (1 = very low, 10 = very high) | Linear scale (1 to 10) | |
| 6 | How would you rate your diet quality? | Multiple choice | Options: `Poor`, `Average`, `Good` (spelled exactly like this) |

## Step 2: Link the form to a Google Sheet

1. Open your form → go to the **Responses** tab.
2. Click the green **Sheets icon** ("Create Spreadsheet").
3. This creates a new Google Sheet that automatically adds a new row
   every time someone submits the form (columns: `Timestamp` + your
   6 question titles).

## Step 3: Publish that Sheet as a CSV link

1. Open the linked Google Sheet.
2. Go to **File → Share → Publish to web**.
3. Under "Link", select the responses sheet (usually named
   `Form Responses 1`).
4. Change the format dropdown to **Comma-separated values (.csv)**.
5. Click **Publish** and confirm.
6. Copy the generated link — it will look something like:
   ```
   https://docs.google.com/spreadsheets/d/e/2PACX-xxxxxxxxxxxxxxxxxxxx/pub?gid=0&single=true&output=csv
   ```

## Step 4: Connect it to the backend script

Open `fetch_and_predict.py` and paste the link into:
```python
SHEET_CSV_URL = "PASTE_YOUR_PUBLISHED_GOOGLE_SHEET_CSV_LINK_HERE"
```

## Step 5: Run the backend

```
python fetch_and_predict.py
```

It will check the sheet every 15 seconds. Fill out your Google Form
(from your phone or another tab) and watch the terminal — within
15 seconds it will print the new response and the predicted stress
level.

## Why this approach (for viva)

- The **Google Form** acts as the front-end / data collection layer —
  no custom UI code needed.
- The **Google Sheet** acts as a live, auto-updating database.
- The **Python backend script** is the only code we wrote: it reads
  new data, feeds it into our trained ML model (`stress_model.pkl`),
  and returns the prediction — the same 6 features and same
  preprocessing steps used during training (`train_model.py`).
