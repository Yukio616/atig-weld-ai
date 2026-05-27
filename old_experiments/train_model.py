import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# LOAD DATASET
df = pd.read_csv("final_atig_dataset.csv")

# REMOVE EMPTY TARGETS
df = df.dropna(subset=["penetration_mm"])

# FILL MISSING VALUES
df["voltage_V"] = df["voltage_V"].fillna(12)

# ENCODE FLUX
encoder = LabelEncoder()

if "fluxes" in df.columns:
    df["flux_encoded"] = encoder.fit_transform(df["fluxes"].astype(str))
else:
    df["flux_encoded"] = 0

# FEATURES
X = df[
    [
        "current_A",
        "voltage_V",
        "travel_speed",
        "flux_encoded"
    ]
]

# TARGET
y = df["penetration_mm"]

# TRAIN MODEL
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# SAVE MODEL
joblib.dump(model, "tig_model.pkl")
joblib.dump(encoder, "flux_encoder.pkl")

print("MODEL TRAINED SUCCESSFULLY")