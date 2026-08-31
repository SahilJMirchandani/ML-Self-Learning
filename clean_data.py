import pandas as pd

# ---------------------------------------------------------
# 1. Load the original UCI Online Retail dataset
# ---------------------------------------------------------

input_file = "data/Online Retail.xlsx"

df = pd.read_excel(input_file)

print("=" * 60)
print("DATA CLEANING - UCI ONLINE RETAIL")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)


# ---------------------------------------------------------
# 2. Remove cancelled invoices
# ---------------------------------------------------------

cancelled = df["InvoiceNo"].astype(str).str.startswith("C")

print("\nCancelled transactions:")
print(cancelled.sum())

df = df[~cancelled].copy()


# ---------------------------------------------------------
# 3. Remove invalid quantities and prices
# ---------------------------------------------------------

before_filter = len(df)

df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

removed_invalid = before_filter - len(df)

print("\nInvalid quantity/price records removed:")
print(removed_invalid)


# ---------------------------------------------------------
# 4. Calculate transaction revenue
# ---------------------------------------------------------

df["Revenue"] = df["Quantity"] * df["UnitPrice"]


# ---------------------------------------------------------
# 5. Extract date from InvoiceDate
# ---------------------------------------------------------

df["Date"] = df["InvoiceDate"].dt.date

df["Date"] = pd.to_datetime(df["Date"])


# ---------------------------------------------------------
# 6. Aggregate transactions into daily sales
# ---------------------------------------------------------

daily_sales = (
    df.groupby("Date", as_index=False)["Revenue"]
    .sum()
    .rename(columns={"Revenue": "Daily_Revenue"})
)


# ---------------------------------------------------------
# 7. Sort chronologically
# ---------------------------------------------------------

daily_sales = daily_sales.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------
# 8. Save processed dataset
# ---------------------------------------------------------

output_file = "data/daily_retail_sales.csv"

daily_sales.to_csv(output_file, index=False)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

print("\nProcessed daily dataset shape:")
print(daily_sales.shape)

print("\nFirst 5 days:")
print(daily_sales.head())

print("\nLast 5 days:")
print(daily_sales.tail())

print("\nDate range:")
print("Start:", daily_sales["Date"].min())
print("End:", daily_sales["Date"].max())

print("\nProcessed dataset saved to:")
print(output_file)

print("\nData cleaning completed successfully.")