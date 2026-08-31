import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. LOAD PREDICTIONS
# =========================================================

results = pd.read_csv(
    "data/random_forest_predictions.csv",
    parse_dates=["Date"]
)


# =========================================================
# 2. CREATE ACTUAL VS PREDICTED PLOT
# =========================================================

plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Revenue"],
    label="Actual Revenue"
)

plt.plot(
    results["Date"],
    results["Predicted_Revenue"],
    label="Predicted Revenue"
)

plt.xlabel("Date")
plt.ylabel("Daily Revenue")

plt.title(
    "Random Forest: Actual vs Predicted Daily Revenue"
)

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()


# =========================================================
# 3. SAVE FIGURE
# =========================================================

plt.savefig(
    "data/actual_vs_predicted.png",
    dpi=300
)

plt.show()

print("Actual vs Predicted graph saved successfully.")
