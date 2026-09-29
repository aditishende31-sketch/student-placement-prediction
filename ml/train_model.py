"""
train_model.py
----------------
Trains four classification models (Logistic Regression, Decision Tree,
Random Forest, KNN), evaluates them, automatically selects the best one
based on F1-score, and saves everything needed for the GUI to use later:

    models/best_model.pkl   -> the winning trained model
    models/scaler.pkl       -> the StandardScaler used to transform inputs
    models/metadata.json    -> comparison table, confusion matrix,
                               feature importance, dataset summary

Why F1-score for selection?
Placement datasets are often imbalanced (more placed than not-placed
students, or vice versa). Accuracy alone can be misleading in that case.
F1-score balances Precision (how many predicted "placed" students really
got placed) and Recall (how many actually-placed students were correctly
identified), making it a fairer single metric for this problem.
"""

import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ml.data_generator import load_or_create_dataset, FEATURE_COLUMNS
from ml.preprocessing import clean_data, split_and_scale
from ml.evaluate_model import evaluate_predictions, get_confusion_matrix_values

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

SELECTION_METRIC = "f1_score"


def get_candidate_models():
    """Returns the four required models with sensible, simple hyperparameters."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }


def save_confusion_matrix_plot(cm_values):
    """Saves a labeled confusion matrix heatmap image."""
    import seaborn as sns

    matrix = np.array([[cm_values["tn"], cm_values["fp"]], [cm_values["fn"], cm_values["tp"]]])
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted: Not Placed", "Predicted: Placed"],
        yticklabels=["Actual: Not Placed", "Actual: Placed"],
    )
    plt.title("Confusion Matrix - Best Model")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()


def save_feature_importance_plot(feature_importance_list):
    """Saves a horizontal bar chart of feature importances, sorted descending."""
    if not feature_importance_list:
        return
    names = [f[0].replace("_", " ") for f in feature_importance_list]
    values = [f[1] for f in feature_importance_list]

    plt.figure(figsize=(7, 6))
    plt.barh(names[::-1], values[::-1], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "feature_importance.png"), dpi=120)
    plt.close()


def train_and_select_best(n_records=800):
    """
    Runs the full training pipeline and saves all required artifacts.
    Returns the metadata dictionary (also written to models/metadata.json).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    df = load_or_create_dataset()
    df_clean, cleaning_report = clean_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df_clean)

    models = get_candidate_models()
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_predictions(y_test, y_pred)
        results[name] = metrics
        trained_models[name] = model

    # Automatically select the best model by F1-score (no hard-coded winner)
    best_model_name = max(results, key=lambda name: results[name][SELECTION_METRIC])
    best_model = trained_models[best_model_name]

    # Confusion matrix for the selected model
    best_predictions = best_model.predict(X_test)
    cm_values = get_confusion_matrix_values(y_test, best_predictions)
    save_confusion_matrix_plot(cm_values)

    # Feature importance is always computed from the Random Forest model
    # (tree-based models expose feature_importances_; this is independent
    # of which model was ultimately selected as "best").
    rf_model = trained_models["Random Forest"]
    importances = rf_model.feature_importances_
    feature_importance_list = sorted(
        zip(FEATURE_COLUMNS, [round(float(v), 4) for v in importances]),
        key=lambda pair: pair[1],
        reverse=True,
    )
    save_feature_importance_plot(feature_importance_list)

    # Save the winning model and the scaler used to preprocess new inputs
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "best_model_name": best_model_name,
        "selection_metric": "F1 Score",
        "model_comparison": results,
        "confusion_matrix": cm_values,
        "feature_importance": feature_importance_list,
        "dataset_summary": {
            "total_records": int(len(df_clean)),
            "placed_count": int((df_clean["Placement"] == 1).sum()),
            "not_placed_count": int((df_clean["Placement"] == 0).sum()),
            "cleaning_report": cleaning_report,
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_metadata():
    """Loads the saved metadata.json (comparison table, confusion matrix, etc.)."""
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def load_model_and_scaler():
    """Loads the saved best model and scaler for making new predictions."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def artifacts_exist():
    return (
        os.path.exists(MODEL_PATH)
        and os.path.exists(SCALER_PATH)
        and os.path.exists(METADATA_PATH)
    )


if __name__ == "__main__":
    result = train_and_select_best()
    print("Best model:", result["best_model_name"])
    print(json.dumps(result["model_comparison"], indent=2))
