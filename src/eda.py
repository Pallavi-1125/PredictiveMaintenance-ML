import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Count failures
print(df["Machine failure"].value_counts())

# Plot graph
df["Machine failure"].value_counts().plot(kind="bar")

plt.title("Machine Failure Distribution")
plt.xlabel("Failure (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.show()