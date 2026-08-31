import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. LOAD DATA
# =========================================================

train = pd.read_csv("data/ml_train.csv")
validation = pd.read_csv("data/ml_validation.csv")
test = pd.read_csv("data/ml_test.csv")


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


# =========================================================
# 3. BEST HYPERPARAMETERS
# =========================================================

best_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)


# =========================================================
# 4. TRAIN FINAL MODEL
# =========================================================

print("=" * 70)
print("FINAL TUNED RANDOM FOREST MODEL")
print("=" * 70)

print("\nTraining final tuned model...")

best_model.fit(X_train, y_train)

print("Training completed successfully.")


# =========================================================
# 5. VALIDATION PERFORMANCE
# =========================================================

validation_predictions = best_model.predict(X_validation)

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


# =========================================================
# 6. TEST PERFORMANCE
# =========================================================

test_predictions = best_model.predict(X_test)

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


# =========================================================
# 7. DISPLAY RESULTS
# =========================================================

print("\n" + "-" * 70)
print("VALIDATION PERFORMANCE")
print("-" * 70)

print(f"MAE  : {validation_mae:.2f}")
print(f"RMSE : {validation_rmse:.2f}")
print(f"R²   : {validation_r2:.4f}")


print("\n" + "-" * 70)
print("FINAL TEST PERFORMANCE")
print("-" * 70)

print(f"MAE  : {test_mae:.2f}")
print(f"RMSE : {test_rmse:.2f}")
print(f"R²   : {test_r2:.4f}")


# =========================================================
# 8. SAMPLE TEST PREDICTIONS
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
# 9. SAVE TEST PREDICTIONS
# =========================================================

results.to_csv(
    "data/final_model_predictions.csv",
    index=False
)

print("\nFinal predictions saved to:")
print("data/final_model_predictions.csv")


# =========================================================
# 10. ACTUAL VS PREDICTED GRAPH
# =========================================================

plt.figure(figsize=(14, 6))

plt.plot(
    test["Date"],
    y_test,
    label="Actual Revenue"
)

plt.plot(
    test["Date"],
    test_predictions,
    label="Predicted Revenue"
)

plt.xlabel("Date")
plt.ylabel("Daily Revenue")

plt.title(
    "Final Tuned Random Forest: Actual vs Predicted Daily Revenue"
)

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.savefig(
    "data/final_model_actual_vs_predicted.png",
    dpi=300
)

plt.show()


print("\nFinal prediction graph saved to:")
print("data/final_model_actual_vs_predicted.png")


print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION COMPLETED")
print("=" * 70)