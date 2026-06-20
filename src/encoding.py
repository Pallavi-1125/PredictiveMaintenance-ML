import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Remove unwanted columns
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

# Encode Type column
encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])

print("Encoded Data:")

print(df.head())

print("\nEncoding Mapping:")
for i, label in enumerate(encoder.classes_):
    print(label, "->", i)