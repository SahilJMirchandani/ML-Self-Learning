import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor


# =========================================================
# 1. LOAD TRAINING DATA
# =========================================================

train = pd.read_csv(
    "data/ml_train.csv"
)


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


# =========================================================
# 3. TRAIN BASELINE RANDOM FOREST
# =========================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# =========================================================
# 4. CALCULATE FEATURE IMPORTANCE
# =========================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


# =========================================================
# 5. DISPLAY FEATURE IMPORTANCE
# =========================================================

print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(index=False)
)


# =========================================================
# 6. CREATE FEATURE IMPORTANCE GRAPH
# =========================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()


# =========================================================
# 7. SAVE GRAPH
# =========================================================

plt.savefig(
    "data/feature_importance.png",
    dpi=300
)

plt.show()


print("\nFeature importance graph saved to:")
print("data/feature_importance.png")