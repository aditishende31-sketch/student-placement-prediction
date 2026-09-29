"""
preprocessing.py
-----------------
Handles data cleaning and preparation before model training:
- missing value check
- duplicate check
- feature/target split
- train/test split
- feature scaling (needed for Logistic Regression and KNN especially)
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.data_generator import FEATURE_COLUMNS, TARGET_COLUMN


def clean_data(df):
    """Basic cleaning: report + drop missing values and duplicate rows."""
    df = df.copy()

    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()

    if missing_count > 0:
        df = df.dropna()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    report = {
        "missing_values_found": int(missing_count),
        "duplicate_rows_found": int(duplicate_count),
        "final_row_count": len(df),
    }
    return df, report


def split_and_scale(df, test_size=0.2, random_state=42):
    """
    Splits the dataset into train/test sets and scales the features.
    Returns X_train, X_test, y_train, y_test, scaler (all scaled arrays are
    numpy arrays; the scaler must be saved so new predictions use the same
    transformation).
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
