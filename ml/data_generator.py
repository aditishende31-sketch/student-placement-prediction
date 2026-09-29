"""
data_generator.py
------------------
Creates a realistic SYNTHETIC dataset for the Student Placement Prediction
project. This is NOT real recruitment data - it is generated using
statistical distributions and a scoring formula so that the 13 input
features have a believable relationship with the final Placement outcome.

Also generates the required Exploratory Data Analysis (EDA) graphs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # allows plotting without a display (safe for first-run setup)
import matplotlib.pyplot as plt
import seaborn as sns

# Folders used by this module (relative paths only - works on any OS)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
DATA_PATH = os.path.join(DATA_DIR, "placement_data.csv")

FEATURE_COLUMNS = [
    "CGPA",
    "10th_Percentage",
    "12th_Percentage",
    "Attendance_Percentage",
    "Backlogs",
    "Internships",
    "Projects",
    "Certifications",
    "Coding_Score",
    "Communication_Score",
    "Aptitude_Score",
    "Technical_Interview_Score",
    "Soft_Skills_Score",
]

TARGET_COLUMN = "Placement"


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_dataset(n_records=800, random_state=42, save=True):
    """
    Generates the synthetic placement dataset.

    The relationship between the features and Placement is built using a
    weighted "placement score" formula (higher CGPA/skills/experience raises
    the score, backlogs/low attendance lower it). Random noise is added so
    the resulting dataset is realistic and not perfectly separable.
    """
    rng = np.random.default_rng(random_state)

    cgpa = np.clip(rng.normal(7.0, 1.0, n_records), 4.0, 10.0)
    tenth = np.clip(rng.normal(78, 9, n_records), 45, 100)
    twelfth = np.clip(rng.normal(74, 10, n_records), 40, 100)
    attendance = np.clip(rng.normal(82, 9, n_records), 45, 100)
    backlogs = np.clip(rng.poisson(0.6, n_records), 0, 8)
    internships = np.clip(rng.poisson(1.0, n_records), 0, 5)
    projects = np.clip(rng.poisson(2.2, n_records), 0, 10)
    certifications = np.clip(rng.poisson(1.3, n_records), 0, 8)
    coding = np.clip(rng.normal(64, 16, n_records), 0, 100)
    communication = np.clip(rng.normal(66, 13, n_records), 0, 100)
    aptitude = np.clip(rng.normal(64, 15, n_records), 0, 100)
    tech_interview = np.clip(rng.normal(60, 16, n_records), 0, 100)
    soft_skills = np.clip(rng.normal(66, 12, n_records), 0, 100)

    # Weighted placement "score" - each weight reflects how strongly that
    # feature is assumed to influence a placement outcome academically.
    score = (
        -0.75
        + 0.70 * (cgpa - 7.0)
        + 0.014 * (tenth - 70)
        + 0.014 * (twelfth - 70)
        + 0.026 * (attendance - 70)
        - 0.55 * backlogs
        + 0.45 * internships
        + 0.30 * projects
        + 0.20 * certifications
        + 0.026 * (coding - 60)
        + 0.017 * (communication - 60)
        + 0.019 * (aptitude - 60)
        + 0.028 * (tech_interview - 60)
        + 0.015 * (soft_skills - 60)
    )

    # Random noise keeps the dataset realistic (not perfectly separable)
    noise = rng.normal(0, 0.9, n_records)
    probability = _sigmoid(score + noise)
    placement = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "CGPA": cgpa.round(2),
            "10th_Percentage": tenth.round(2),
            "12th_Percentage": twelfth.round(2),
            "Attendance_Percentage": attendance.round(2),
            "Backlogs": backlogs.astype(int),
            "Internships": internships.astype(int),
            "Projects": projects.astype(int),
            "Certifications": certifications.astype(int),
            "Coding_Score": coding.round(2),
            "Communication_Score": communication.round(2),
            "Aptitude_Score": aptitude.round(2),
            "Technical_Interview_Score": tech_interview.round(2),
            "Soft_Skills_Score": soft_skills.round(2),
            "Placement": placement.astype(int),
        }
    )

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    return df


def generate_eda_plots(df):
    """
    Creates the required Exploratory Data Analysis graphs and saves them
    to the visualizations/ folder.
    """
    os.makedirs(VIZ_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_specs = [
        ("CGPA", "cgpa_vs_placement.png", "CGPA vs Placement"),
        ("Attendance_Percentage", "attendance_vs_placement.png", "Attendance % vs Placement"),
        ("Coding_Score", "coding_vs_placement.png", "Coding Score vs Placement"),
        ("Internships", "internships_vs_placement.png", "Number of Internships vs Placement"),
    ]

    for column, filename, title in plot_specs:
        plt.figure(figsize=(6, 4.5))
        if column == "Internships":
            # discrete count feature -> bar-style comparison works better
            sns.barplot(
                x=column, y=TARGET_COLUMN, data=df,
                hue=column, legend=False, palette="viridis",
            )
            plt.ylabel("Placement Rate (0 = Not Placed, 1 = Placed)")
        else:
            sns.boxplot(
                x=TARGET_COLUMN, y=column, data=df,
                hue=TARGET_COLUMN, legend=False, palette="Set2",
            )
            plt.xlabel("Placement (0 = Not Placed, 1 = Placed)")
            plt.ylabel(column.replace("_", " "))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, filename), dpi=120)
        plt.close()

    # Correlation heatmap across all numeric columns
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True)
    plt.title("Correlation Heatmap - All Features")
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()


def load_or_create_dataset():
    """
    Loads the dataset from disk if it already exists, otherwise generates
    a brand-new one (and its EDA plots) and returns it.
    """
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    df = generate_dataset()
    generate_eda_plots(df)
    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    generate_eda_plots(dataset)
    print(f"Generated {len(dataset)} records at {DATA_PATH}")
    print(dataset["Placement"].value_counts())
