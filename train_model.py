import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATA
# =========================================================

train = pd.read_csv(
    "data/ml_train.csv",
    parse_dates=["Date"]
)

validation = pd.read_csv(
    "data/ml_validation.csv",
    parse_dates=["Date"]
)

test = pd.read_csv(
    "data/ml_test.csv",
    parse_dates=["Date"]
)


print("=" * 70)
print("RANDOM FOREST REGRESSION")
print("=" * 70)

print("\nDataset sizes:")
print("Training:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))


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

X_test = test[features]
y_test = test[target]


print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget:")
print("-", target)


# =========================================================
# 3. CREATE RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 4. TRAIN MODEL
# =========================================================

print("\n" + "-" * 70)
print("TRAINING MODEL")
print("-" * 70)

model.fit(X_train, y_train)

print("\nModel training completed successfully.")


# =========================================================
# 5. VALIDATION PREDICTIONS
# =========================================================

validation_predictions = model.predict(X_validation)


# =========================================================
# 6. VALIDATION METRICS
# =========================================================

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions
) ** 0.5

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)


print("\n" + "-" * 70)
print("VALIDATION PERFORMANCE")
print("-" * 70)

print(f"\nMAE  : {validation_mae:.2f}")
print(f"RMSE : {validation_rmse:.2f}")
print(f"R²   : {validation_r2:.4f}")


# =========================================================
# 7. TEST PREDICTIONS
# =========================================================

test_predictions = model.predict(X_test)


# =========================================================
# 8. TEST METRICS
# =========================================================

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\n" + "-" * 70)
print("FINAL TEST PERFORMANCE")
print("-" * 70)

print(f"\nMAE  : {test_mae:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"R²   : {test_r2:.4f}")


# =========================================================
# 9. DISPLAY SAMPLE PREDICTIONS
# =========================================================

results = pd.DataFrame({
    "Date": test["Date"],
    "Actual_Revenue": y_test,
    "Predicted_Revenue": test_predictions
})


print("\n" + "-" * 70)
print("SAMPLE TEST PREDICTIONS")
print("-" * 70)

print(results.head(10).to_string(index=False))


# =========================================================
# 10. SAVE PREDICTIONS
# =========================================================

results.to_csv(
    "data/random_forest_predictions.csv",
    index=False
)

print("\nPredictions saved to:")
print("data/random_forest_predictions.csv")


print("\n" + "=" * 70)
print("RANDOM FOREST EXPERIMENT COMPLETED")
print("=" * 70)