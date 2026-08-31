import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------
# 1. Load the processed daily sales dataset
# ---------------------------------------------------------

file_path = "data/daily_retail_sales.csv"

df = pd.read_csv(file_path, parse_dates=["Date"])

# Make sure the data is sorted chronologically
df = df.sort_values("Date").reset_index(drop=True)

print("=" * 70)
print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print("\nTotal observations:")
print(len(df))

print("\nOverall date range:")
print(df["Date"].min(), "to", df["Date"].max())


# ---------------------------------------------------------
# 2. Calculate chronological split sizes
# ---------------------------------------------------------

n = len(df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
validation = df.iloc[train_end:validation_end].copy()
test = df.iloc[validation_end:].copy()


# ---------------------------------------------------------
# 3. Display split information
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("CHRONOLOGICAL SPLIT")
print("-" * 70)

print("\nTraining Set:")
print("Number of observations:", len(train))
print("Start:", train["Date"].min())
print("End:", train["Date"].max())

print("\nValidation Set:")
print("Number of observations:", len(validation))
print("Start:", validation["Date"].min())
print("End:", validation["Date"].max())

print("\nTest Set:")
print("Number of observations:", len(test))
print("Start:", test["Date"].min())
print("End:", test["Date"].max())


# ---------------------------------------------------------
# 4. Verify chronological ordering
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("CHRONOLOGICAL ORDER VERIFICATION")
print("-" * 70)

print(
    "\nTraining ends before validation starts:",
    train["Date"].max() < validation["Date"].min()
)

print(
    "Validation ends before test starts:",
    validation["Date"].max() < test["Date"].min()
)


# ---------------------------------------------------------
# 5. Save the three datasets
# ---------------------------------------------------------

train.to_csv("data/train.csv", index=False)
validation.to_csv("data/validation.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("\n" + "-" * 70)
print("FILES SAVED")
print("-" * 70)

print("\ndata/train.csv")
print("data/validation.csv")
print("data/test.csv")


# ---------------------------------------------------------
# 6. Demonstrate TimeSeriesSplit
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TIMESERIES SPLIT")
print("=" * 70)

# We use the training data for time-series cross-validation.
tscv = TimeSeriesSplit(n_splits=5)

print("\nNumber of TimeSeriesSplit folds:", tscv.n_splits)

for fold, (train_index, validation_index) in enumerate(
    tscv.split(train), start=1
):
    fold_train = train.iloc[train_index]
    fold_validation = train.iloc[validation_index]

    print("\nFold", fold)

    print(
        "Training:",
        len(fold_train),
        "observations |",
        fold_train["Date"].min(),
        "to",
        fold_train["Date"].max()
    )

    print(
        "Validation:",
        len(fold_validation),
        "observations |",
        fold_validation["Date"].min(),
        "to",
        fold_validation["Date"].max()
    )

print("\n" + "=" * 70)
print("TEMPORAL SPLITTING COMPLETED SUCCESSFULLY")
print("=" * 70)