import pandas as pd
import boto3
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.pyfunc
import numpy as np
import joblib


# =========================
# S3 CONFIG
# =========================

s3 = boto3.client("s3")

BUCKET = "mlops-house-prediction2"
KEY = "processed/2026-09-30/Mlops_house_prediction_clean_v1.csv"


# =========================
# FETCH DATA FROM S3
# =========================

def fetch_data():
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)

    df = pd.read_csv(
        StringIO(obj["Body"].read().decode("utf-8"))
    )

    return df


df = fetch_data()

print(f"Fetched shape: {df.shape}")


# =========================
# FEATURES / TARGET
# =========================

FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
    "garage",
    "location_score"
]

X = df[FEATURES]
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# MLFLOW
# =========================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("mlops-house-prediction2")


# =========================
# PYFUNC MODEL
# =========================

class HousePriceModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = joblib.load(
            context.artifacts["model"]
        )

    def predict(self, context, model_input):

        return self.model.predict(model_input)


# =========================
# TRAINING
# =========================

with mlflow.start_run() as run:

    n_estimators = 150
    max_depth = 8

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)


    # =========================
    # PREDICTION
    # =========================

    preds = model.predict(X_test)


    # =========================
    # METRICS
    # =========================

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )


    # =========================
    # MLFLOW PARAMETERS
    # =========================

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    mlflow.log_param(
        "data_source",
        f"s3://{BUCKET}/{KEY}"
    )


    # =========================
    # MLFLOW METRICS
    # =========================

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)


    # =========================
    # SAVE MODEL
    # =========================

    joblib.dump(
        model,
        "model.pkl"
    )


    # =========================
    # REGISTER MODEL
    # =========================

    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=HousePriceModel(),
        artifacts={
            "model": "model.pkl"
        },
        registered_model_name="house-price-predictor"
    )


    print(
        f"\nMAE: {mae:.2f}"
        f" | RMSE: {rmse:.2f}"
        f" | R2: {r2:.4f}"
    )

    print(
        f"\nRun ID: {run.info.run_id}"
    )

    print(
        f"Model URI: {model_info.model_uri}"
    )

    print(
        "\nModel registered successfully!"
    )