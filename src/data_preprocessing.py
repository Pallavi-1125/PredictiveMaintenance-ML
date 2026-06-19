import pandas as pd

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Show first 5 rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Basic information
print(df.info())