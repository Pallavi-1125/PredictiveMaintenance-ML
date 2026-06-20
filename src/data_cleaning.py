import pandas as pd

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Remove unnecessary columns
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

print("Remaining Columns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())