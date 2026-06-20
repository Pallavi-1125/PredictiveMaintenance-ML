import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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

# Encode categorical column
encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])

# Features and target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Save model
joblib.dump(model, "models/predictive_maintenance_model.pkl")

print("Model saved successfully")