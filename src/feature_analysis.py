import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Select numeric columns
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Correlation with machine failure
correlation = numeric_df.corr()["Machine failure"]

print(correlation)

# Plot correlation
correlation.plot(kind="bar")

plt.title("Feature Correlation with Machine Failure")
plt.ylabel("Correlation Value")
plt.show()