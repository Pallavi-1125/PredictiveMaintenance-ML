import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/ai4i_predictive_maintenance.csv")

# Drop unnecessary columns
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

# Features and target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTarget Distribution:")
print(y.value_counts())