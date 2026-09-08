# Mental Health & Stress Level Predictor

A machine learning classification project that predicts a person's stress level
(**Low / Medium / High**) from their daily lifestyle routine — sleep hours,
study/work hours, screen time, physical activity, pressure score, and diet
quality.

## Project Documents
- [Project Proposal](Mental_Health_Stress_Predictor_Proposal.docx)
- [Phase 2 Technical Blueprint](2_Phase2_Technical_Blueprint.md)
- [Viva Preparation Guide](3_Viva_Preparation_Guide.md)

## Code
See [code/README_HowToRun.md](code/README_HowToRun.md) for setup and run
instructions, and [code/GOOGLE_FORM_SETUP.md](code/GOOGLE_FORM_SETUP.md) for
setting up the Google Form input.

```
code/
├── generate_sample_data.py   # creates a sample dataset
├── train_model.py            # cleans data, does EDA, trains & saves the model
├── fetch_and_predict.py      # backend: reads Google Sheet responses, predicts
├── app.py                    # optional local Streamlit form
├── dataset_analyzer_app.py   # upload any CSV, train a Decision Tree, get a full report
└── requirements.txt
```

### Dataset Analyzer (upload any dataset)
`dataset_analyzer_app.py` is a general-purpose tool: upload any CSV dataset
(e.g. from GitHub/Kaggle), pick the target column and feature columns, and it
trains a Decision Tree Classifier and shows test accuracy, a classification
report, a confusion matrix, an actual-vs-predicted table, and a picture of
the trained decision tree.
```
streamlit run code/dataset_analyzer_app.py
```

## Author
Shaikh Haris Uddin — IU04-0320-0277
