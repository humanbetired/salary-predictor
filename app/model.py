import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

mlflow.set_tracking_uri(
    "file:///D:/AI Journey/project_01/salary_predictor/mlruns"
)

_model = None
_encoder = None


def _build_encoder():
    """Rebuild TargetEncoder dari training data."""
    df = pd.read_csv("data/ml_salaries.csv")
    df = df[(df['salary_in_usd'] >= 20000) & (df['salary_in_usd'] <= 500000)]
    top_countries = ['US', 'GB', 'CA', 'ES', 'DE']
    df = df[df['company_location'].isin(top_countries)].copy()

    df['is_fully_remote'] = (df['remote_ratio'] == 100).astype(int)
    df['is_large_company'] = (df['company_size'] == 'L').astype(int)
    df['is_senior'] = df['experience_level'].isin(['SE', 'EX']).astype(int)
    df['target'] = np.log1p(df['salary_in_usd'])

    feature_cols = [
        'experience_level', 'employment_type', 'remote_ratio',
        'company_size', 'employee_residence', 'company_location',
        'work_year', 'is_fully_remote', 'is_large_company', 'is_senior'
    ]
    cat_cols = [
        'experience_level', 'employment_type',
        'company_size', 'employee_residence', 'company_location'
    ]

    X = df[feature_cols]
    y = df['target']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    te = TargetEncoder(cols=cat_cols)
    te.fit(X_train, y_train)
    return te


def load_model():
    """Load model dari MLflow Model Registry."""
    global _model, _encoder
    if _model is None:
        print("Loading model dari MLflow registry...")
        _model = mlflow.sklearn.load_model("models:/salary-predictor-v1/1")
        _encoder = _build_encoder()
        print("Model loaded!")
    return _model, _encoder


def predict(features: dict) -> float:
    """Terima dict fitur, return prediksi gaji dalam USD."""
    model, encoder = load_model()

    df = pd.DataFrame([features])

    # Derived features
    df['is_fully_remote'] = (df['remote_ratio'] == 100).astype(int)
    df['is_large_company'] = (df['company_size'] == 'L').astype(int)
    df['is_senior'] = df['experience_level'].isin(['SE', 'EX']).astype(int)

    feature_cols = [
        'experience_level', 'employment_type', 'remote_ratio',
        'company_size', 'employee_residence', 'company_location',
        'work_year', 'is_fully_remote', 'is_large_company', 'is_senior'
    ]

    df_enc = encoder.transform(df[feature_cols])
    pred_log = model.predict(df_enc)
    pred_usd = float(np.expm1(pred_log)[0])

    return round(pred_usd, 2)