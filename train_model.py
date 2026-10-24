import os
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# Create models directory
os.makedirs("models", exist_ok=True)


def train_model(name, X, y, target_names=None):
    print(f"\nTraining {name} model...")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create ML pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name} Accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

    # Save model and its input information
    bundle = {
        "model": model,
        "features": list(X.columns),
        "classes": [str(c) for c in model.classes_],
        "target_names": target_names or {}
    }

    joblib.dump(bundle, f"models/{name}_model.pkl")

    print(f"{name} model saved successfully!")


def load_csv_dataset(path, target_candidates):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Please download the dataset."
        )

    df = pd.read_csv(path)

    # Remove whitespace from column names
    df.columns = df.columns.str.strip()

    # Find target column
    target_column = next(
        (
            col for col in target_candidates
            if col.lower() in [c.lower() for c in df.columns]
        ),
        None
    )

    if target_column is None:
        raise ValueError(
            f"Target column not found in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    # Match actual column capitalization
    target_column = next(
        c for c in df.columns
        if c.lower() == target_column.lower()
    )

    df = df.dropna(subset=[target_column])

    # Keep numeric input features
    X = df.drop(columns=[target_column]).select_dtypes(
        include="number"
    )

    y = df[target_column]

    # Remove rows with non-numeric or missing target labels
    valid = y.notna()

    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if X.empty:
        raise ValueError(f"No numeric features found in {path}.")

    if y.nunique() < 2:
        raise ValueError(
            f"{path} must contain at least two target classes."
        )

    return X, y


# ==========================================
# 1. HEART DISEASE MODEL
# ==========================================

def train_heart():
    X, y = load_csv_dataset(
        "dataset/heart.csv",
        ["target", "HeartDisease", "heart_disease", "num"]
    )

    train_model(
        "heart",
        X,
        y,
        target_names={
            "0": "No heart disease",
            "1": "Possible heart disease"
        }
    )


# ==========================================
# 2. DIABETES MODEL
# ==========================================

def train_diabetes():
    X, y = load_csv_dataset(
        "dataset/diabetes.csv",
        ["Outcome", "outcome", "diabetes"]
    )

    train_model(
        "diabetes",
        X,
        y,
        target_names={
            "0": "No diabetes",
            "1": "Possible diabetes"
        }
    )


# ==========================================
# 3. BREAST CANCER MODEL
# ==========================================

def train_breast_cancer():
    # Use Scikit-learn's built-in dataset
    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    # Original dataset:
    # 0 = malignant
    # 1 = benign
    y = pd.Series(data.target)

    train_model(
        "cancer",
        X,
        y,
        target_names={
            "0": "Malignant class",
            "1": "Benign class"
        }
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    try:
        train_heart()
    except Exception as e:
        print(f"\nHeart model error: {e}")

    try:
        train_diabetes()
    except Exception as e:
        print(f"\nDiabetes model error: {e}")

    try:
        train_breast_cancer()
    except Exception as e:
        print(f"\nBreast cancer model error: {e}")

    print("\nModel training process completed.")