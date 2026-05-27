import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# LOAD CLEANED DATASET
# ==========================================================

df = pd.read_csv("final_atig_dataset.csv")

print("\nDataset Loaded Successfully\n")
print(df.head())

# ==========================================================
# REMOVE BAD ROWS
# ==========================================================

df = df.dropna(
    subset=[
        "current_A",
        "penetration_mm"
    ]
)

# ==========================================================
# FILL MISSING VOLTAGE
# ==========================================================

df["voltage_V"] = df["voltage_V"].fillna(12)

# ==========================================================
# FILL MISSING SPEED
# ==========================================================

df["travel_speed"] = df["travel_speed"].fillna(
    df["travel_speed"].median()
)

# ==========================================================
# CLEAN FLUX NAMES
# ==========================================================

df["fluxes"] = df["fluxes"].fillna("No Flux")

df["fluxes"] = (
    df["fluxes"]
    .str.replace("Tio2", "TiO2")
    .str.replace("tio2", "TiO2")
    .str.replace("sio2", "SiO2")
)

# ==========================================================
# ENCODE FLUXES
# ==========================================================

encoder = LabelEncoder()

df["flux_encoded"] = encoder.fit_transform(
    df["fluxes"]
)

# ==========================================================
# CALCULATE HEAT INPUT
# ==========================================================

# Heat Input = (V * I) / Speed

df["heat_input"] = (
    df["voltage_V"] *
    df["current_A"]
) / df["travel_speed"]

# ==========================================================
# FEATURES + TARGET
# ==========================================================

X = df[
    [
        "current_A",
        "voltage_V",
        "travel_speed",
        "heat_input",
        "flux_encoded"
    ]
]

y = df["penetration_mm"]

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================================
# RANDOM FOREST MODEL
# ==========================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================================
# PREDICTION
# ==========================================================

predictions = model.predict(X_test)

# ==========================================================
# EVALUATION
# ==========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nMAE  : {mae:.3f}")
print(f"R²   : {r2:.3f}")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n")
print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(importance)

# ==========================================================
# PLOT FEATURE IMPORTANCE
# ==========================================================

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.title(
    "Random Forest Feature Importance"
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig("feature_importance.png")

# ==========================================================
# SAVE CLEANED DATASET
# ==========================================================

df.to_csv(
    "cleaned_final_dataset.csv",
    index=False
)

print("\n")
print("=" * 60)
print("FILES SAVED")
print("=" * 60)

print("\nSaved:")
print("1. cleaned_final_dataset.csv")
print("2. feature_importance.png")

# ==========================================================
# SAMPLE PREDICTION
# ==========================================================

print("\n")
print("=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

sample = pd.DataFrame({
    "current_A": [200],
    "voltage_V": [12],
    "travel_speed": [150],
    "heat_input": [(12 * 200) / 150],
    "flux_encoded": [1]
})

prediction = model.predict(sample)

print(
    f"\nPredicted Penetration = "
    f"{prediction[0]:.3f} mm"
)