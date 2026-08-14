# How to Run This Project

## 1. Install requirements
```
pip install -r requirements.txt
```

## 2. Generate the sample dataset (skip this once you have the real Kaggle CSV)
```
python generate_sample_data.py
```
This creates `data/stress_dataset.csv`.

> To use a real Kaggle dataset instead: download it, rename/save it as
> `data/stress_dataset.csv`, and make sure its columns match the names
> used in `train_model.py` (`Sleep_Hours`, `Study_Work_Hours`,
> `Screen_Time`, `Physical_Activity`, `Pressure_Score`, `Diet_Quality`,
> `Stress_Level`) — rename columns in the CSV if needed.

## 3. Train the model
```
python train_model.py
```
This prints EDA info + model accuracy, saves 3 PNG graphs, and saves:
- `stress_model.pkl`
- `scaler.pkl`
- `diet_encoder.pkl`
- `target_encoder.pkl`

## 4. Set up the Google Form input (main flow)
Follow `GOOGLE_FORM_SETUP.md` to create the Google Form + linked
Google Sheet, then paste the published CSV link into
`fetch_and_predict.py`.

Run the backend:
```
python fetch_and_predict.py
```
It checks the Google Sheet every 15 seconds. Submit the Google Form
(from a phone or another tab) and the terminal will print the new
response and the predicted stress level.

## (Optional) Alternative: simple web form instead of Google Form
If you'd rather not use Google Forms, `app.py` (Streamlit) gives the
same prediction through a basic local web form:
```
streamlit run app.py
```

## Folder structure
```
code/
├── data/
│   └── stress_dataset.csv        (created by generate_sample_data.py)
├── generate_sample_data.py       (creates sample dataset)
├── train_model.py                (cleans data, does EDA, trains & saves model)
├── fetch_and_predict.py          (main backend: reads Google Sheet, predicts, prints result)
├── GOOGLE_FORM_SETUP.md          (steps to create the Google Form + Sheet)
├── app.py                        (optional: Streamlit form + prediction)
├── requirements.txt
└── stress_model.pkl, scaler.pkl, diet_encoder.pkl, target_encoder.pkl
    (created after running train_model.py)
```
