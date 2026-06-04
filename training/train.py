import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
mlflow.set_tracking_uri("file:///D:/AI Journey/project_01/salary_predictor/mlruns")
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')




def load_and_preprocess(data_path: str):
    """Load, filter, dan preprocess dataset."""
    df = pd.read_csv(data_path)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    level_weights = {'EN': 3.0, 'MI': 1.5, 'SE': 1.0, 'EX': 4.0}
    sample_weights = X_train['experience_level'].map(level_weights).values

    te = TargetEncoder(cols=cat_cols)
    X_train_enc = te.fit_transform(X_train, y_train)
    X_test_enc = te.transform(X_test)

    return X_train_enc, X_test_enc, y_train, y_test, sample_weights, te, X_train


def evaluate(model, X_test_enc, y_test, sample_weights=None):
    """Hitung MAE dan R2 dalam USD."""
    y_pred_log = model.predict(X_test_enc)
    y_pred_usd = np.expm1(y_pred_log)
    y_test_usd = np.expm1(y_test)

    mae = mean_absolute_error(y_test_usd, y_pred_usd)
    r2 = r2_score(y_test_usd, y_pred_usd)
    return mae, r2


def run_experiment(run_name: str, model, params: dict,
                   X_train_enc, X_test_enc, y_train, y_test,
                   sample_weights, te):
    """Jalankan satu eksperimen dan log semua ke MLflow."""

    mlflow.set_experiment("salary-predictor")

    with mlflow.start_run(run_name=run_name):
        if isinstance(model, XGBRegressor):
            model.fit(X_train_enc, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train_enc, y_train)

        mae, r2 = evaluate(model, X_test_enc, y_test)

        mlflow.log_params(params)

        mlflow.log_metric("mae_usd", mae)
        mlflow.log_metric("r2_score", r2)

        if isinstance(model, XGBRegressor):
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        print(f"[{run_name}] MAE: ${mae:,.0f} | R2: {r2:.3f}")

        return mlflow.active_run().info.run_id


if __name__ == "__main__":
    print("Loading data...")
    X_train_enc, X_test_enc, y_train, y_test, sample_weights, te, X_train = \
        load_and_preprocess("data/ml_salaries.csv")

    run_experiment(
        run_name="rf-baseline",
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        params={"model": "RandomForest", "n_estimators": 100},
        X_train_enc=X_train_enc, X_test_enc=X_test_enc,
        y_train=y_train, y_test=y_test,
        sample_weights=sample_weights, te=te
    )

    run_experiment(
        run_name="rf-deeper",
        model=RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42),
        params={"model": "RandomForest", "n_estimators": 300, "max_depth": 15},
        X_train_enc=X_train_enc, X_test_enc=X_test_enc,
        y_train=y_train, y_test=y_test,
        sample_weights=sample_weights, te=te
    )

    run_experiment(
        run_name="gbm-baseline",
        model=GradientBoostingRegressor(n_estimators=200, random_state=42),
        params={"model": "GradientBoosting", "n_estimators": 200},
        X_train_enc=X_train_enc, X_test_enc=X_test_enc,
        y_train=y_train, y_test=y_test,
        sample_weights=sample_weights, te=te
    )

    run_experiment(
        run_name="xgb-baseline",
        model=XGBRegressor(n_estimators=300, max_depth=5,
                           learning_rate=0.05, random_state=42, verbosity=0),
        params={"model": "XGBoost", "n_estimators": 300,
                "max_depth": 5, "learning_rate": 0.05},
        X_train_enc=X_train_enc, X_test_enc=X_test_enc,
        y_train=y_train, y_test=y_test,
        sample_weights=sample_weights, te=te
    )

    run_experiment(
        run_name="xgb-tuned",
        model=XGBRegressor(
            subsample=0.9, n_estimators=700, min_child_weight=5,
            max_depth=5, learning_rate=0.01, colsample_bytree=0.9,
            random_state=42, verbosity=0
        ),
        params={"model": "XGBoost", "n_estimators": 700, "max_depth": 5,
                "learning_rate": 0.01, "subsample": 0.9,
                "colsample_bytree": 0.9, "min_child_weight": 5},
        X_train_enc=X_train_enc, X_test_enc=X_test_enc,
        y_train=y_train, y_test=y_test,
        sample_weights=sample_weights, te=te
    )

    print("\n✅ Semua eksperimen selesai.")
    print("Jalankan: mlflow ui")
    print("Buka browser: http://127.0.0.1:5000")