import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


# ============================================================
# MLflow Experiment Tracking - Retail Sales Forecasting
# ============================================================

print("=" * 70)
print("MLFLOW EXPERIMENT TRACKING")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------

train = pd.read_csv("data/ml_train.csv")
validation = pd.read_csv("data/ml_validation.csv")
test = pd.read_csv("data/ml_test.csv")

features = [
    "Day_of_Week",
    "Day_of_Month",
    "Month",
    "Week_of_Year",
    "Lag_1",
    "Lag_7",
    "Lag_14",
    "Rolling_7_Day_Mean"
]

target = "Daily_Revenue"

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]


# ------------------------------------------------------------
# 2. Set MLflow experiment
# ------------------------------------------------------------

mlflow.set_experiment("Retail Sales Forecasting")


# ------------------------------------------------------------
# 3. Start MLflow run
# ------------------------------------------------------------

with mlflow.start_run(run_name="Tuned Random Forest"):

    # Tuned hyperparameters obtained previously
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "random_state": 42
    }

    # Log parameters
    mlflow.log_params(params)

    # --------------------------------------------------------
    # 4. Train final model
    # --------------------------------------------------------

    model = RandomForestRegressor(**params)

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # 5. Validation predictions
    # --------------------------------------------------------

    validation_predictions = model.predict(X_validation)

    validation_mae = mean_absolute_error(
        y_validation,
        validation_predictions
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_predictions
        )
    )

    validation_r2 = r2_score(
        y_validation,
        validation_predictions
    )

    # --------------------------------------------------------
    # 6. Test predictions
    # --------------------------------------------------------

    test_predictions = model.predict(X_test)

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    # --------------------------------------------------------
    # 7. Log metrics
    # --------------------------------------------------------

    mlflow.log_metrics({
        "validation_mae": validation_mae,
        "validation_rmse": validation_rmse,
        "validation_r2": validation_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    })

    # --------------------------------------------------------
    # 8. Log model
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model"
    )

    # --------------------------------------------------------
    # 9. Display results
    # --------------------------------------------------------

    print("\nMLflow run completed successfully.")

    print("\nLogged Parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    print("\nValidation Metrics:")
    print(f"MAE  : {validation_mae:.2f}")
    print(f"RMSE : {validation_rmse:.2f}")
    print(f"R²   : {validation_r2:.4f}")

    print("\nTest Metrics:")
    print(f"MAE  : {test_mae:.2f}")
    print(f"RMSE : {test_rmse:.2f}")
    print(f"R²   : {test_r2:.4f}")

    print("\nModel logged to MLflow successfully.")


print("\n" + "=" * 70)
print("MLFLOW TRACKING COMPLETED")
print("=" * 70)