# Viva Preparation & Defense Guide
## Mental Health & Stress Level Predictor

---

**Q1. What is the problem your project solves?**
People often don't realize how their daily habits — sleep, screen time, study/work hours, activity, diet — add up to stress until it becomes serious. Our project takes these lifestyle inputs from a simple form and predicts whether the person's stress level is Low, Medium, or High, so they can catch it early and adjust their habits.

**Q2. Why is this a classification problem and not regression?**
Because the output (Stress Level) is a category — Low, Medium, or High — not a continuous number. Regression predicts a number (like exact stress score 0–100); classification predicts a label/class. Since we only need the category, classification fits.

**Q3. Why did you choose Random Forest / Logistic Regression as your models?**
- **Logistic Regression** is used as a simple, interpretable baseline — it's fast, easy to explain, and works well when relationships between features and the outcome are roughly linear.
- **Random Forest** is an ensemble of many decision trees; each tree votes, and the majority vote wins. It's used because it usually handles non-linear relationships and mixed feature types better, is less sensitive to outliers, and generally gives more robust accuracy than a single model — without needing heavy tuning.
We trained both and compared their accuracy, then picked whichever performed better on the test set (in our run, Logistic Regression slightly outperformed Random Forest — this can vary depending on the dataset).

**Q4. Why do you split data into train and test sets?**
To check whether the model actually "learned" patterns, or just memorized the data. We train on 80% of the data, then test on the remaining 20% the model has never seen. If accuracy on the test set is close to accuracy on the training set, the model is generalizing well, not overfitting.

**Q5. What is `random_state=42` and why do you use it?**
It's a seed value that makes random operations (like the train/test split) reproducible — every time you re-run the code, you get the exact same split and same results, instead of a different random split each time. Any number can be used; 42 is just a common convention.

**Q6. What is `stratify=y` doing in your train_test_split?**
It makes sure the proportion of Low/Medium/High classes is roughly the same in both the training set and the test set. Without it, by random chance, the test set might end up with very few examples of one class, making evaluation unreliable.

**Q7. Why do you scale your features (StandardScaler)?**
Features like Sleep Hours (0–12) and Pressure Score (1–10) are on different numeric ranges. Some models (especially Logistic Regression, which is distance/weight based) can be biased toward features with larger raw values. Scaling converts every feature to a similar range (mean 0, standard deviation 1) so no feature unfairly dominates just because of its scale.

**Q8. Why do you encode categorical variables like Diet Quality?**
Machine learning models work with numbers, not text. `LabelEncoder` converts text categories ("Poor", "Average", "Good") into numbers (e.g., 0, 1, 2) so the model can use them mathematically. We must apply the exact same encoder (fit on training data) to any new input, otherwise the numbers won't mean the same thing.

**Q9. How does the model process the form input in your project?**
1. The user submits their routine through a **Google Form**.
2. Google Form automatically saves the response as a new row in a linked **Google Sheet**, which is published to the web as a live CSV link.
3. Our backend script (`fetch_and_predict.py`) periodically reads that CSV and detects the new row.
4. It converts the categorical input (Diet Quality) using the saved encoder.
5. It arranges all inputs into the same column order used during training.
6. It applies the same saved `StandardScaler` to scale the input.
7. It calls `model.predict()` on the scaled input, which returns an encoded number.
8. It converts that number back into a readable label (Low/Medium/High) using the saved target encoder, and prints it as the result.

*(We also kept a basic optional Streamlit form, `app.py`, as an alternative local interface — same preprocessing and prediction steps, just triggered by a button click instead of a sheet check.)*

**Q10. What evaluation metrics did you use and why?**
- **Accuracy** — overall percentage of correct predictions.
- **Precision** — of all predictions for a class, how many were actually correct (important to avoid false alarms).
- **Recall** — of all actual cases of a class, how many did we correctly catch (important so we don't miss high-stress cases).
- **F1-score** — balance between precision and recall.
- **Confusion Matrix** — shows exactly which classes get confused with which (e.g., Medium being predicted as High), which is more informative than accuracy alone, especially for a 3-class problem.

**Q11. What would you do if the model's accuracy is low?**
- Collect more/better quality data.
- Try feature engineering (e.g., combine screen time + study hours into a "digital load" feature).
- Try other models (SVM, Gradient Boosting/XGBoost).
- Tune hyperparameters (e.g., `n_estimators`, `max_depth` for Random Forest) using `GridSearchCV`.
- Check for class imbalance and use `class_weight='balanced'` or oversampling (e.g., SMOTE).

**Q12. Is this project using Deep Learning or Computer Vision?**
No. This is a classical machine learning project on structured/tabular data using scikit-learn. There are no images, and no neural networks are required — the relationships between lifestyle habits and stress can be captured well with simpler models like Logistic Regression and Random Forest.

**Q13. What is the real-world limitation of this project?**
The model is trained on self-reported survey data, so it reflects correlations, not medical diagnoses. It should be treated as an awareness/screening tool, not a substitute for professional mental health assessment. Also, stress is subjective and influenced by many factors beyond the 6 lifestyle features used here.

**Q14. How would you extend this project in the future?**
- Add more features (e.g., financial stress, social support, caffeine intake).
- Track a user's inputs over time (trend analysis) instead of a single snapshot.
- Deploy it as a proper web/mobile app with user accounts.
- Add a recommendation engine suggesting specific lifestyle changes based on which feature is driving the "High" prediction (using feature importance / SHAP values).

---

### Quick "Elevator Pitch" (memorize this one)
"We built a machine learning model that predicts a person's stress level — Low, Medium, or High — based on their daily lifestyle habits like sleep, study/work hours, screen time, physical activity, pressure levels, and diet. We used a Kaggle survey dataset, cleaned and encoded it, trained and compared Logistic Regression and Random Forest classifiers, and wrapped the best model in a simple Streamlit form so anyone can enter their routine and instantly see their predicted stress level."
