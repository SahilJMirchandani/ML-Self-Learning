import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATA
# =========================================================

train = pd.read_csv("data/ml_train.csv")
validation = pd.read_csv("data/ml_validation.csv")


# =========================================================
# 2. DEFINE FEATURES AND TARGET
# =========================================================

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


# =========================================================
# 3. TIME SERIES CROSS-VALIDATION
# =========================================================

tscv = TimeSeriesSplit(n_splits=5)


# =========================================================
# 4. RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 5. HYPERPARAMETER GRID
# =========================================================

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2, 4]
}


# =========================================================
# 6. GRID SEARCH
# =========================================================

print("=" * 70)
print("RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 70)

print("\nUsing 5-fold TimeSeriesSplit...")
print("Searching for the best Random Forest parameters...\n")

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)


# =========================================================
# 7. BEST PARAMETERS
# =========================================================

print("\n" + "-" * 70)
print("BEST HYPERPARAMETERS")
print("-" * 70)

print(grid_search.best_params_)

print("\nBest cross-validation MAE:")
print(f"{-grid_search.best_score_:.2f}")


# =========================================================
# 8. BEST MODEL
# =========================================================

best_model = grid_search.best_estimator_


# =========================================================
# 9. VALIDATION PREDICTIONS
# =========================================================

validation_predictions = best_model.predict(X_validation)


# =========================================================
# 10. VALIDATION METRICS
# =========================================================

mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

rmse = mean_squared_error(
    y_validation,
    validation_predictions
) ** 0.5

r2 = r2_score(
    y_validation,
    validation_predictions
)


print("\n" + "-" * 70)
print("TUNED MODEL - VALIDATION PERFORMANCE")
print("-" * 70)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# =========================================================
# 11. BASELINE COMPARISON
# =========================================================

print("\n" + "-" * 70)
print("BASELINE VS TUNED MODEL")
print("-" * 70)

print("\nBaseline Validation:")
print("MAE  : 11832.75")
print("RMSE : 17825.08")
print("R²   : 0.3743")

print("\nTuned Validation:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# =========================================================
# 12. SAVE TUNED VALIDATION PREDICTIONS
# =========================================================

results = pd.DataFrame({
    "Date": validation["Date"],
    "Actual_Revenue": y_validation,
    "Predicted_Revenue": validation_predictions
})

results.to_csv(
    "data/tuned_validation_predictions.csv",
    index=False
)


print("\nTuned validation predictions saved to:")
print("data/tuned_validation_predictions.csv")

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING COMPLETED")
print("=" * 70)