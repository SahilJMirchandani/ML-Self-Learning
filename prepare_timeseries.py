import pandas as pd

# ---------------------------------------------------------
# 1. Load cleaned daily sales
# ---------------------------------------------------------

input_file = "data/daily_retail_sales.csv"

df = pd.read_csv(input_file, parse_dates=["Date"])

df = df.sort_values("Date").reset_index(drop=True)

print("=" * 70)
print("PREPARING CONTINUOUS DAILY TIME SERIES")
print("=" * 70)

print("\nOriginal daily observations:")
print(len(df))

print("Original date range:")
print(df["Date"].min(), "to", df["Date"].max())


# ---------------------------------------------------------
# 2. Create a complete daily date range
# ---------------------------------------------------------

complete_dates = pd.date_range(
    start=df["Date"].min(),
    end=df["Date"].max(),
    freq="D"
)


# ---------------------------------------------------------
# 3. Reindex the dataset to include every calendar day
# ---------------------------------------------------------

df = (
    df.set_index("Date")
    .reindex(complete_dates)
    .rename_axis("Date")
    .reset_index()
)


# ---------------------------------------------------------
# 4. Replace missing revenue with zero
# ---------------------------------------------------------

df["Daily_Revenue"] = df["Daily_Revenue"].fillna(0)


# ---------------------------------------------------------
# 5. Display results
# ---------------------------------------------------------

print("\nContinuous daily observations:")
print(len(df))

print("\nMissing revenue values after filling:")
print(df["Daily_Revenue"].isnull().sum())

print("\nNumber of zero-sales days:")
print((df["Daily_Revenue"] == 0).sum())

print("\nFirst 10 records:")
print(df.head(10))

print("\nLast 10 records:")
print(df.tail(10))


# ---------------------------------------------------------
# 6. Save continuous time series
# ---------------------------------------------------------

output_file = "data/continuous_daily_sales.csv"

df.to_csv(output_file, index=False)

print("\nContinuous dataset saved to:")
print(output_file)

print("\nContinuous time series preparation completed successfully.")