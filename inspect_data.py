import pandas as pd

# Path to the UCI Online Retail dataset
file_path = "data/Online Retail.xlsx"

# Load the Excel dataset
df = pd.read_excel(file_path)

print("=" * 60)
print("UCI ONLINE RETAIL DATASET")
print("=" * 60)

# Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Columns:")
print(df.columns.tolist())

# First five records
print("\n3. First 5 Records:")
print(df.head())

# Data types
print("\n4. Data Types:")
print(df.dtypes)

# Missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Date information
print("\n6. Date Range:")
print("Earliest:", df["InvoiceDate"].min())
print("Latest:", df["InvoiceDate"].max())

# Unique values
print("\n7. Unique Invoices:")
print(df["InvoiceNo"].nunique())

print("\n8. Unique Customers:")
print(df["CustomerID"].nunique())

print("\nInspection completed successfully.")