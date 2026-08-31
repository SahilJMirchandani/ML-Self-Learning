import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/model_comparison.csv")

models = df["Model"].tolist()
validation_r2 = df["Validation_R2"].tolist()
test_r2 = df["Test_R2"].tolist()

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, validation_r2, width, label="Validation R²")
plt.bar(x + width/2, test_r2, width, label="Test R²")

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("Baseline vs Tuned Random Forest Performance")

plt.xticks(x, models)
plt.ylim(0, 0.5)

plt.legend()
plt.tight_layout()
plt.show()