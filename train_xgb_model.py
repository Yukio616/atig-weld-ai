import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI-Based TIG / A-TIG Weld Prediction",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# LIGHT BLUE / WHITE THEME
# =========================================================

st.markdown("""
<style>

/* WHOLE PAGE */

.stApp {
    background-color: #eef5fc;
    color: #0B3C6D;
}

/* MAIN CONTAINER */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 5%;
    padding-right: 5%;
}

/* TITLES */

h1 {
    color: #0B3C6D !important;
    font-size: 48px !important;
    font-weight: 800 !important;
}

h2 {
    color: #145DA0 !important;
    font-weight: 700 !important;
}

h3 {
    color: #145DA0 !important;
}

p {
    color: #1B1B1B;
    font-size: 17px;
}

/* HERO BOX */

.hero-box {

    background: linear-gradient(
        135deg,
        #145DA0,
        #2E8BC0
    );

    padding: 35px;

    border-radius: 20px;

    color: white;

    box-shadow: 0px 6px 20px rgba(0,0,0,0.15);
}

/* METRICS */

[data-testid="stMetric"] {

    background-color: white;

    border-radius: 16px;

    padding: 20px;

    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);

    border: 1px solid #d7e7f7;
}

/* BUTTONS */

.stButton>button {

    background: linear-gradient(
        90deg,
        #145DA0,
        #2E8BC0
    );

    color: white;

    border-radius: 12px;

    border: none;

    padding: 12px 24px;

    font-size: 16px;

    font-weight: bold;
}

/* SLIDERS */

.stSlider > div > div > div > div {
    background-color: #2E8BC0 !important;
}

/* SELECT BOX */

div[data-baseweb="select"] {

    background-color: white !important;

    border-radius: 10px !important;

    border: 1px solid #cfe2f3 !important;
}

/* TABLES */

table {

    background-color: white;

    border-radius: 12px;

    overflow: hidden;
}

/* REMOVE STREAMLIT DARK EFFECTS */

header {
    background-color: transparent !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* MOBILE */

@media (max-width: 768px) {

    h1 {
        font-size: 30px !important;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("tig_model.pkl")
encoder = joblib.load("flux_encoder.pkl")

# =========================================================
# HEADER WITH LOGOS
# =========================================================

col1, col2, col3 = st.columns([1,4,1])

with col1:
    st.image("assets/CEG_col1.png", width=120)

with col2:

    st.markdown("""
    <div class="hero-box">

    <h1 style='text-align:center;color:white;'>

    AI-Based TIG / A-TIG Weld Prediction

    </h1>

    <h3 style='text-align:center;color:white;'>

    Nano Oxide Activated Fluxes for SS304 Welding

    </h3>

    <p style='text-align:center;color:white;'>

    Interactive AI-assisted weld penetration prediction system.

    </p>

    </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("assets/iiti.png", width=120)

# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("---")

st.header("📘 Project Overview")

st.markdown("""

### Conventional TIG Welding Challenges

- Limited penetration in thick sections
- Low depth-to-width ratio
- Higher distortion and heat input
- Multi-pass welding requirement

### A-TIG Solution

Activated nano oxide fluxes:
- reverse Marangoni convection,
- constrict the arc,
- improve penetration depth,
- reduce bead width.

""")

# =========================================================
# LITERATURE SECTION
# =========================================================

st.markdown("---")

st.header("📚 Literature Survey")

col1, col2 = st.columns(2)

with col1:

    try:
        st.image(
            "assets/literature/lit1.png",
            caption="Cross-sections of A-TIG welds"
        )

    except:
        st.info("Add image: assets/literature/lit1.png")

with col2:

    try:
        st.image(
            "assets/literature/lit2.png",
            caption="Flux effect on weld penetration"
        )

    except:
        st.info("Add image: assets/literature/lit2.png")

# =========================================================
# WELDING PARAMETERS TABLE
# =========================================================

st.markdown("---")

st.header("⚙ Recommended Welding Parameters")

param_df = pd.DataFrame({

    "Parameter": [

        "Base Material",
        "Current",
        "Voltage",
        "Travel Speed",
        "Shielding Gas",
        "Flux Thickness",
        "Particle Size",
        "Welding Process"
    ],

    "Value": [

        "SS304",
        "110–140 A",
        "10–14 V",
        "100 mm/min",
        "100% Argon",
        "20–30 µm",
        "20–50 nm",
        "DCEN A-TIG"
    ]
})

st.table(param_df)

# =========================================================
# INPUT PARAMETERS
# =========================================================

st.markdown("---")

st.header("⚙ Welding Parameters")

left, right = st.columns(2)

with left:

    current = st.slider(
        "Current (A)",
        90,
        160,
        120
    )

    voltage = st.slider(
        "Voltage (V)",
        9,
        15,
        12
    )

with right:

    travel_speed = st.slider(
        "Travel Speed (mm/min)",
        80,
        200,
        120
    )

    flux = st.selectbox(
        "Flux Type",
        [

            "No Flux",
            "Fe2O3",
            "Al2O3",
            "Cr2O3",
            "TiO2",
            "SiO2",
            "TiO2+SiO2",
            "TiO2+Al2O3",
            "SiO2+Cr2O3"
        ]
    )

# =========================================================
# MATERIAL
# =========================================================

material = st.selectbox(
    "Material",
    [
        "SS304",
        "SS316",
        "Duplex 2205"
    ]
)

# =========================================================
# ENCODE FLUX
# =========================================================

try:
    flux_encoded = encoder.transform([flux])[0]

except:
    flux_encoded = 0

# =========================================================
# MODEL INPUT
# =========================================================

input_df = pd.DataFrame({

    "current_A": [current],
    "voltage_V": [voltage],
    "travel_speed": [travel_speed],
    "flux_encoded": [flux_encoded]
})

# =========================================================
# MODEL PREDICTION
# =========================================================

prediction = model.predict(input_df)[0]

# =========================================================
# ENGINEERING BASED CORRECTION
# =========================================================

flux_factor = {

    "No Flux": 0.80,

    "Fe2O3": 1.00,

    "Al2O3": 1.10,

    "Cr2O3": 1.20,

    "TiO2": 1.60,

    "SiO2": 1.85,

    "TiO2+SiO2": 2.00,

    "TiO2+Al2O3": 1.75,

    "SiO2+Cr2O3": 1.90
}

prediction *= flux_factor.get(flux, 1.0)

# =========================================================
# MATERIAL EFFECT
# =========================================================

if material == "SS304":
    prediction *= 1.00

elif material == "SS316":
    prediction *= 0.92

elif material == "Duplex 2205":
    prediction *= 1.12

# =========================================================
# LIMIT PREDICTION
# =========================================================

prediction = max(1.0, prediction)
prediction = min(14.0, prediction)

# =========================================================
# RELATIVE PENETRATION
# =========================================================

baseline = 3.0

relative_penetration = (
    (prediction - baseline)
    / baseline
) * 100

# =========================================================
# HEAT INPUT
# =========================================================

heat_input = (
    voltage * current * 60
) / (1000 * travel_speed)

# =========================================================
# RESULTS
# =========================================================

st.markdown("---")

st.header("📊 Prediction Results")

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Penetration Depth",
        f"{prediction:.2f} mm"
    )

with m2:

    st.metric(
        "Relative Increase",
        f"{relative_penetration:.1f}%"
    )

with m3:

    st.metric(
        "Heat Input",
        f"{heat_input:.2f} kJ/mm"
    )

# =========================================================
# AI INTERPRETATION
# =========================================================

st.markdown("---")

st.header("🤖 AI Interpretation")

if prediction < 4:

    st.warning("""

    Conventional TIG morphology expected.

    • Wide shallow weld bead
    • Lower penetration
    • Weak arc constriction

    """)

elif prediction < 8:

    st.info("""

    Moderate A-TIG penetration behavior predicted.

    • Improved depth-to-width ratio
    • Moderate Marangoni reversal
    • Improved weld penetration

    """)

else:

    st.success("""

    Strong A-TIG behavior predicted.

    • Deep narrow penetration
    • Strong arc constriction
    • Enhanced Marangoni convection reversal

    """)

# =========================================================
# MARANGONI EXPLANATION
# =========================================================

st.markdown("---")

st.header("🌊 Marangoni Convection Mechanism")

st.markdown("""

Activated fluxes modify:
- oxygen concentration,
- surface tension gradients,
- weld pool flow behavior.

This reverses Marangoni convection:
- outward ➜ inward flow.

Result:
- deeper penetration,
- narrower weld bead,
- improved weld morphology.

""")

# =========================================================
# WELD MORPHOLOGY VISUALIZATION
# =========================================================

st.markdown("---")

st.header("🧪 Predicted Weld Morphology")

fig, ax = plt.subplots(figsize=(4,4))

# PLATE

ax.plot([-5,5],[0,0], linewidth=6)

# =========================================================
# SHAPE LOGIC
# =========================================================

depth = prediction / 2

if flux == "No Flux":

    width = 4.5

elif flux in ["Fe2O3", "Al2O3"]:

    width = 3.5

elif flux == "Cr2O3":

    width = 2.8

elif flux in ["TiO2", "SiO2"]:

    width = 1.8

else:

    width = 1.5

# =========================================================
# WELD SHAPE
# =========================================================

points = np.array([

    [-width, 0],

    [-width/2, -depth*0.3],

    [0, -depth],

    [width/2, -depth*0.3],

    [width, 0]
])

polygon = Polygon(
    points,
    closed=True,
    alpha=0.7
)

ax.add_patch(polygon)

# =========================================================
# GRAPH SETTINGS
# =========================================================

ax.set_xlim(-5,5)

ax.set_ylim(-8,2)

ax.set_title(
    "Predicted Weld Cross Section",
    fontsize=14
)

ax.set_xlabel("Weld Width")

ax.set_ylabel("Penetration Depth")

ax.grid(True)

st.pyplot(fig)

# =========================================================
# PENETRATION TREND
# =========================================================

st.markdown("---")

st.header("📈 Literature Penetration Trend")

trend_df = pd.DataFrame({

    "Flux": [

        "No Flux",
        "Fe2O3",
        "Al2O3",
        "Cr2O3",
        "TiO2",
        "SiO2"
    ],

    "Relative Penetration": [

        1.0,
        1.2,
        1.4,
        1.8,
        2.3,
        2.6
    ]
})

st.bar_chart(
    trend_df.set_index("Flux")
)

# =========================================================
# FLUX EFFECTS
# =========================================================

st.markdown("---")

st.header("🧪 Flux Influence")

flux_info = {

    "SiO2":
    "Highest penetration due to strong arc constriction.",

    "TiO2":
    "Deep penetration and stable weld morphology.",

    "Fe2O3":
    "Moderate penetration enhancement.",

    "Al2O3":
    "Moderate arc constriction behavior.",

    "Cr2O3":
    "Improved penetration characteristics."
}

for key, value in flux_info.items():

    st.info(f"### {key}\n{value}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""

### AI-Assisted Welding Engineering Platform

Developed for:
- A-TIG Welding
- Weld Penetration Prediction
- Metallurgical Visualization
- AI + Materials Engineering Integration

Anna University | IIT Indore

""")