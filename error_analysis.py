import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("=" * 70)
print("FINAL MODEL ERROR ANALYSIS")
print("=" * 70)

# Load final predictions
df = pd.read_csv("data/final_model_predictions.csv")

# Calculate errors
df["Error"] = df["Actual_Revenue"] - df["Predicted_Revenue"]
df["Absolute_Error"] = abs(df["Error"])

# RMSE
rmse = np.sqrt(mean_squared_error(
    df["Actual_Revenue"],
    df["Predicted_Revenue"]
))

# MAE
mae = mean_absolute_error(
    df["Actual_Revenue"],
    df["Predicted_Revenue"]
)

print("\nOverall Error Metrics")
print("-" * 70)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

print("\nLargest Prediction Errors")
print("-" * 70)

largest_errors = df.sort_values(
    "Absolute_Error",
    ascending=False
).head(10)

print(
    largest_errors[
        [
            "Date",
            "Actual_Revenue",
            "Predicted_Revenue",
            "Error",
            "Absolute_Error"
        ]
    ].to_string(index=False)
)

# Save error analysis
df.to_csv(
    "data/final_error_analysis.csv",
    index=False
)

print("\nError analysis saved to:")
print("data/final_error_analysis.csv")


# ------------------------------------------------------------
# ACTUAL VS PREDICTED
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Actual_Revenue"],
    label="Actual Revenue"
)

plt.plot(
    df["Date"],
    df["Predicted_Revenue"],
    label="Predicted Revenue"
)

plt.xlabel("Date")
plt.ylabel("Daily Revenue")
plt.title("Final Random Forest: Actual vs Predicted Revenue")

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Predicted_Revenue"],
    df["Error"]
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Revenue")
plt.ylabel("Prediction Error")
plt.title("Final Random Forest Residual Analysis")

plt.tight_layout()

plt.show()

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)