# Student Placement Prediction Using Machine Learning

A 3rd-year college Statistical Machine Learning (SMLD) mini-project: a
desktop application that predicts whether a student is likely to be
placed, using a trained classification model and a Tkinter GUI.

## Problem Statement

Campus placement outcomes depend on many factors - academics, technical
skills, communication ability, and interview performance. Manually
judging "placement readiness" is subjective. This project explores
whether a machine learning classification model can learn patterns from
historical-style student data and predict placement likelihood in a
consistent, data-driven way.

## Objective

Given 13 academic/skill/experience features for a student, the
application predicts:

- **Placement Status** - `PLACED` or `NOT PLACED`
- **Placement Probability** - e.g. `86.4%`, taken directly from the
  trained model's `predict_proba()` output (never randomly generated)

The prediction is an **academic ML prediction** and does **not**
guarantee actual employment or placement outcomes.

## Features Implemented

- Synthetic dataset generation (500-1000+ records) with realistic,
  noisy relationships between features and placement
- Data cleaning: missing value and duplicate checks
- Exploratory Data Analysis (5 saved graphs)
- Train/test split with feature scaling (`StandardScaler`)
- 4 trained and compared ML models: Logistic Regression, Decision Tree,
  Random Forest, K-Nearest Neighbors
- Automatic best-model selection based on F1-score (no hard-coded winner)
- Saved model + scaler (`joblib`), reloaded instead of retrained on
  every launch
- Confusion matrix (TP/TN/FP/FN) and feature-importance graphs
- Tkinter desktop GUI with 5 sections: Dashboard, Predict Placement,
  Prediction History, Model Performance, Feature Importance / About
- Full input validation with friendly error messages
- SQLite-backed prediction history with auto-generated Student IDs
  (ST001, ST002, ...)
- General, non-guaranteeing improvement suggestions after each prediction
- Automatic first-time setup (dataset -> training -> database -> GUI)
- Error handling throughout so the GUI never crashes on bad input or
  missing files

## Technologies

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Tkinter,
SQLite, Joblib.

## Machine Learning Algorithms

- **Logistic Regression** - a simple linear model that estimates the
  probability of a binary outcome (Placed / Not Placed). Fast, easy to
  interpret, and a good baseline for classification problems.
- **Decision Tree** - splits data into branches based on feature
  thresholds (e.g. "CGPA > 7.5?"). Easy to visualize and explain, but
  can overfit if allowed to grow too deep (depth is limited here).
- **Random Forest** - trains many decision trees on random subsets of
  data/features and averages their votes. Usually more accurate and
  stable than a single decision tree, and also gives feature importance.
- **K-Nearest Neighbors (KNN)** - classifies a student by looking at the
  K most similar students (by scaled feature distance) in the training
  data and taking a majority vote. Requires feature scaling to work well
  since it relies on distance calculations.

## Dataset

`data/placement_data.csv` is **synthetically generated** for this
academic project - it is **not real recruitment data** from any
institution or company. It is created using statistical distributions
(normal/Poisson) for each feature, combined into a weighted "placement
score" with random noise added, so the resulting relationships are
realistic (higher CGPA/skills generally help, backlogs generally hurt)
but not perfectly clean or 100% predictable.

### Features Used (13 total)

1. CGPA (0-10)
2. 10th Percentage (0-100)
3. 12th/Diploma Percentage (0-100)
4. Attendance Percentage (0-100)
5. Number of Backlogs
6. Number of Internships
7. Number of Projects Completed
8. Number of Certifications
9. Coding Score (0-100)
10. Communication Score (0-100)
11. Aptitude Score (0-100)
12. Technical Interview Score (0-100)
13. Soft Skills Score (0-100)

## ML Workflow

```
Synthetic Dataset
      -> Data Cleaning (missing values, duplicates)
      -> Train/Test Split (80/20, stratified)
      -> Feature Scaling (StandardScaler)
      -> Train 4 Models
      -> Evaluate (Accuracy, Precision, Recall, F1)
      -> Automatically Select Best Model (highest F1-score)
      -> Save Model + Scaler (joblib)
      -> GUI loads saved model for new predictions
```

## Evaluation Metrics

- **Accuracy** - proportion of all predictions that were correct.
- **Precision** - of students predicted "Placed", how many actually were.
- **Recall** - of students who were actually "Placed", how many did the
  model correctly find.
- **F1 Score** - the harmonic mean of Precision and Recall; used here as
  the model-selection metric because it balances both concerns, which
  matters when the dataset is imbalanced (more placed than not-placed
  students, or vice versa).
- **Confusion Matrix** - a table of True Positives, True Negatives,
  False Positives, and False Negatives for the selected model.

## Project Structure

```
student_placement_prediction/
|-- app.py                  Entry point; runs first-time setup, launches GUI
|-- requirements.txt
|-- README.md
|-- VIVA.md
|
|-- data/
|   `-- placement_data.csv        Generated synthetic dataset
|
|-- models/
|   |-- best_model.pkl            Saved winning model
|   |-- scaler.pkl                Saved StandardScaler
|   `-- metadata.json             Comparison table, confusion matrix, feature importance
|
|-- database/
|   `-- predictions.db            SQLite prediction history
|
|-- ml/
|   |-- data_generator.py         Synthetic dataset + EDA plots
|   |-- preprocessing.py          Cleaning, split, scaling
|   |-- train_model.py            Trains 4 models, selects best, saves artifacts
|   `-- evaluate_model.py         Shared metric calculation helpers
|
|-- gui/
|   |-- main_window.py            Navigation + window layout
|   |-- dashboard.py              Dashboard tab
|   |-- prediction_form.py        Predict Placement tab
|   |-- history.py                Prediction History tab
|   |-- performance.py            Model Performance tab
|   `-- feature_importance.py     Feature Importance / About tab
|
|-- utils/
|   |-- validation.py             Input validation rules
|   |-- database.py               SQLite helper functions
|   `-- image_viewer.py           Cross-platform "open image" helper
|
`-- visualizations/
    |-- cgpa_vs_placement.png
    |-- attendance_vs_placement.png
    |-- coding_vs_placement.png
    |-- internships_vs_placement.png
    |-- correlation_heatmap.png
    |-- confusion_matrix.png
    `-- feature_importance.png
```

## How to Install (Windows)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Tkinter and SQLite ship with standard Python installs. On some Linux
> distributions, install Tkinter separately if needed:
> `sudo apt-get install python3-tk`

## How to Run

```bash
python app.py
```

On the very first run, the app will automatically generate the dataset,
train and compare all 4 models, save the best one, generate all
visualization graphs, and create the SQLite database - this takes a few
seconds. On every later run, it loads the saved model directly.

## How to Use the Application

1. **Dashboard** - see dataset size, placement split, the selected best
   model, its accuracy, and how many predictions you've made so far.
2. **Predict Placement** - fill in all 13 fields and click
   **PREDICT PLACEMENT** to see the result, probability, and improvement
   suggestions. Click **Clear Form** to reset.
3. **Prediction History** - view every past prediction in a table.
   Use **Refresh** to reload, or **Clear History** to delete all records.
4. **Model Performance** - see the 4-model comparison table and the
   confusion matrix for the selected model (button to open the image).
5. **Feature Importance / About** - see which features mattered most to
   the model, and general information about the project.

## Common Errors and Fixes

| Problem | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'tkinter'` | Tkinter not installed with your Python | Reinstall Python from python.org (Windows/Mac), or `sudo apt-get install python3-tk` (Linux) |
| App seems to freeze for a few seconds on first run | It's training 4 ML models and generating graphs | Wait - this only happens once |
| "Please enter a valid CGPA..." | Invalid/empty/out-of-range input | Correct the highlighted field and try again |
| Confusion matrix / feature importance image won't open | Default image viewer not configured on your OS | Open the PNG manually from `visualizations/` |
| Want to regenerate everything from scratch | Old dataset/model/db present | Delete the `data/`, `models/`, and `database/` folders, then run `python app.py` again |

## Limitations

- The dataset is entirely **synthetic** and generated for academic
  purposes - it does not represent any real institution's students.
- The model reflects patterns in this synthetic data only and cannot
  account for real-world factors like market conditions, company-specific
  hiring criteria, or interview variability.
- Feature importance shows what the model relied on statistically; it
  does **not** prove that a feature *causes* placement or non-placement.

## Future Improvements

- Train on real, anonymized institutional placement data (with consent)
- Add more features (e.g. specific skill certifications, resume score)
- Deploy as a web application for wider accessibility
- Hyperparameter tuning (GridSearchCV) for better model performance
- Larger, more diverse dataset across multiple academic years

## Author

*[Your Name Here]*
*[Your Roll Number / Class Here]*
*[Your College Name Here]*
