
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("synthetic_ss304_atig_dataset.csv")

# Encode flux names
encoder = LabelEncoder()
df["flux_encoded"] = encoder.fit_transform(df["flux"])

# Features
X = df[
    [
        "current_A",
        "voltage_V",
        "travel_speed_mm_min",
        "heat_input",
        "plate_thickness_mm",
        "flux_encoded"
    ]
]

# Target
y = df["penetration_mm"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"R2 Score: {r2:.3f}")
print(f"MAE: {mae:.3f}")

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance:\n")
print(importance.sort_values(by="Importance", ascending=False))
