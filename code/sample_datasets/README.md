# Sample Datasets (for demo / showing report to teacher)

These were downloaded using `download_sample_datasets.py` (real datasets
from Kaggle, via the `kagglehub` library — no manual searching needed).

Upload any of these into `dataset_analyzer_app.py`, pick the **Target
column** listed below, then click "Train Decision Tree Model".

| File | Target column to pick | Rows | Tested Accuracy (depth=6) | What it predicts |
|---|---|---|---|---|
| mental_health_dataset.csv | `treatment` | 292,364 | 73.7% | Whether a person seeks mental health treatment |
| credit_card_fraud.csv | `Class` | 284,807 | 99.9% | Whether a transaction is fraud (0/1) — very imbalanced |
| airline_passenger_satisfaction.csv | `satisfaction` | 25,976 | 91.8% | Whether a passenger was satisfied |
| adult_census_income.csv | `income` | 32,561 | 85.4% | Whether income is above/below $50K |
| telco_customer_churn.csv | `Churn` | 7,043 | 79.2% | Whether a customer will leave (churn) |
| stroke_prediction.csv | `stroke` | 5,110 | 94.0% | Whether a person had a stroke — very imbalanced |
| heart_disease.csv | `target` | 1,025 | 88.3% | Whether a person has heart disease |
| employee_attrition.csv | `Attrition` | 1,470 | 82.3% | Whether an employee leaves the company |
| mushroom_classification.csv | `class` | 8,124 | 99.9% | Whether a mushroom is edible or poisonous |
| titanic.csv | `Survived` | 891 | 74.8% | Whether a passenger survived (drop Name/Ticket/Cabin/PassengerId columns in the app) |
| bank_marketing.csv | `y` | 4,521 | 90.5% | Whether a customer subscribes to a term deposit |

**Note:** `titanic.csv` has some columns (`Name`, `Ticket`, `Cabin`,
`PassengerId`) that are just identifiers, not useful patterns — in the
app's "feature columns" box, unselect them for a cleaner tree and better
accuracy.

**Not downloaded:** the Pima Indians Diabetes dataset needs a logged-in
Kaggle account (private/gated resource) — skipped automatically.

**Tip for your demo:** `mushroom_classification.csv` and
`credit_card_fraud.csv` give the highest, most "impressive" accuracy
(~99.9%), while `telco_customer_churn.csv` or `titanic.csv` give more
realistic, explainable numbers (~75-80%) if your teacher asks why
accuracy isn't 100% — a good chance to explain overfitting vs.
realistic noisy real-world data.
