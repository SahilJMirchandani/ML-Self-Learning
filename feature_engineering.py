import pandas as pd

# ---------------------------------------------------------
# 1. Load continuous daily time series
# ---------------------------------------------------------

input_file = "data/continuous_daily_sales.csv"

df = pd.read_csv(input_file, parse_dates=["Date"])

df = df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------
# 2. Create calendar features
# ---------------------------------------------------------

df["Day_of_Week"] = df["Date"].dt.dayofweek
df["Day_of_Month"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Week_of_Year"] = df["Date"].dt.isocalendar().week.astype(int)


# ---------------------------------------------------------
# 3. Create historical lag features
# ---------------------------------------------------------

df["Lag_1"] = df["Daily_Revenue"].shift(1)
df["Lag_7"] = df["Daily_Revenue"].shift(7)
df["Lag_14"] = df["Daily_Revenue"].shift(14)


# ---------------------------------------------------------
# 4. Create previous 7-day rolling average
# ---------------------------------------------------------

df["Rolling_7_Day_Mean"] = (
    df["Daily_Revenue"]
    .shift(1)
    .rolling(window=7)
    .mean()
)


# ---------------------------------------------------------
# 5. Remove rows without sufficient history
# ---------------------------------------------------------

df = df.dropna().reset_index(drop=True)


# ---------------------------------------------------------
# 6. Save final feature-engineered dataset
# ---------------------------------------------------------

output_file = "data/final_feature_engineered_sales.csv"

df.to_csv(output_file, index=False)


# ---------------------------------------------------------
# 7. Display results
# ---------------------------------------------------------

print("=" * 70)
print("FINAL FEATURE ENGINEERING")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nFeatures:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFinal feature-engineered dataset saved to:")
print(output_file)

print("\nFeature engineering completed successfully.")