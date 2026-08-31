import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------
# 1. Load final feature-engineered dataset
# ---------------------------------------------------------

file_path = "data/final_feature_engineered_sales.csv"

df = pd.read_csv(file_path, parse_dates=["Date"])

# Ensure chronological order
df = df.sort_values("Date").reset_index(drop=True)

print("=" * 70)
print("FINAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print("\nTotal observations:")
print(len(df))

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())


# ---------------------------------------------------------
# 2. Chronological 70:15:15 split
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
print("Observations:", len(train))
print("Start:", train["Date"].min())
print("End:", train["Date"].max())

print("\nValidation Set:")
print("Observations:", len(validation))
print("Start:", validation["Date"].min())
print("End:", validation["Date"].max())

print("\nTest Set:")
print("Observations:", len(test))
print("Start:", test["Date"].min())
print("End:", test["Date"].max())


# ---------------------------------------------------------
# 4. Verify chronological ordering
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TEMPORAL ORDER VERIFICATION")
print("-" * 70)

print(
    "\nTrain < Validation:",
    train["Date"].max() < validation["Date"].min()
)

print(
    "Validation < Test:",
    validation["Date"].max() < test["Date"].min()
)


# ---------------------------------------------------------
# 5. Save the final ML datasets
# ---------------------------------------------------------

train.to_csv("data/ml_train.csv", index=False)
validation.to_csv("data/ml_validation.csv", index=False)
test.to_csv("data/ml_test.csv", index=False)

print("\n" + "-" * 70)
print("FILES SAVED")
print("-" * 70)

print("\ndata/ml_train.csv")
print("data/ml_validation.csv")
print("data/ml_test.csv")


# ---------------------------------------------------------
# 6. TimeSeriesSplit on training data
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TIMESERIES CROSS-VALIDATION")
print("=" * 70)

tscv = TimeSeriesSplit(n_splits=5)

print("\nNumber of folds:", tscv.n_splits)

for fold, (train_index, validation_index) in enumerate(
    tscv.split(train), start=1
):

    fold_train = train.iloc[train_index]
    fold_validation = train.iloc[validation_index]

    print("\nFold", fold)

    print(
        "Training:",
        len(fold_train),
        "|",
        fold_train["Date"].min(),
        "to",
        fold_train["Date"].max()
    )

    print(
        "Validation:",
        len(fold_validation),
        "|",
        fold_validation["Date"].min(),
        "to",
        fold_validation["Date"].max()
    )


print("\n" + "=" * 70)
print("FINAL TEMPORAL SPLITTING COMPLETED SUCCESSFULLY")
print("=" * 70)