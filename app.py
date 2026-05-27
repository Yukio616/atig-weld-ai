import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI TIG Weld Prediction",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color:#020617;
    color:white;
    font-family:'Segoe UI';
}

.block-container{
    padding-top:1rem;
    padding-left:3rem;
    padding-right:3rem;
}

h1,h2,h3,h4{
    color:#7dd3fc;
}

.stSelectbox div[data-baseweb="select"]{
    background:#111827;
    border-radius:15px;
    color:white;
}

.stSlider > div > div{
    color:#38bdf8;
}

.metric-card{
    background:linear-gradient(135deg,#111827,#1e293b);
    padding:25px;
    border-radius:20px;
    border:1px solid #334155;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,0.3);
}

.glow{
    color:#7dd3fc;
    text-shadow:0px 0px 15px #38bdf8;
}

.slide-card{
    background:#0f172a;
    padding:20px;
    border-radius:25px;
    border:1px solid #334155;
    box-shadow:0px 4px 20px rgba(56,189,248,0.15);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

col1, col2, col3 = st.columns([1,4,1])

with col1:
    try:
        st.image("assets/CEG_col1.png", width=120)
    except:
        st.empty()

with col2:

    st.markdown("""
    <div style="
    background:linear-gradient(135deg,#111827,#1e293b);
    border-radius:30px;
    padding:50px;
    text-align:center;
    border:1px solid #374151;
    box-shadow:0px 8px 25px rgba(0,0,0,0.3);
    ">

    <h1 style="
    font-size:65px;
    color:white;
    margin-bottom:15px;
    ">
    AI-Based TIG / A-TIG<br>
    Weld Prediction
    </h1>

    <h2 style="
    font-size:24px;
    color:#7dd3fc;
    ">
    AI-Assisted Smart Welding Platform
    </h2>

    <p style="
    color:#d1d5db;
    font-size:20px;
    line-height:2;
    ">

    Machine Learning • Nano Oxide Fluxes • A-TIG Welding • SS304 Research

    <br>

    Anna University • IIT Indore

    </p>

    <p style="
    color:#94a3b8;
    font-size:18px;
    ">
    Developed by Harsha Varthanan
    </p>

    </div>
    """, unsafe_allow_html=True)

with col3:
    try:
        st.image("assets/iiti.png", width=120)
    except:
        st.empty()

# =========================================================
# INTRO
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="
background:#0f172a;
padding:30px;
border-radius:25px;
border:1px solid #334155;
">

<h2 class="glow">AI-Assisted Welding Metallurgy Platform</h2>

<p style="font-size:20px; line-height:2; color:#d1d5db;">

This platform combines welding metallurgy,
nano oxide activated fluxes,
machine learning prediction,
and penetration trend visualization
for intelligent A-TIG weld analysis.

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# SLIDES
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 class="glow">
📘 Literature Survey Slides
</h1>
""", unsafe_allow_html=True)

slide_files = [
    "papers/slide1.png",
    "papers/slide2.png",
    "papers/slide3.png",
    "papers/slide4.png",
    "papers/slide5.png",
    "papers/slide6.png",
    "papers/slide7.png",
    "papers/slide8.png",
    "papers/slide9.png",
    "papers/slide10.png",
    "papers/slide11.png"
]

if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

c1, c2, c3 = st.columns([1,8,1])

with c1:
    if st.button("⬅️"):
        st.session_state.slide_index -= 1

        if st.session_state.slide_index < 0:
            st.session_state.slide_index = len(slide_files)-1

with c3:
    if st.button("➡️"):
        st.session_state.slide_index += 1

        if st.session_state.slide_index >= len(slide_files):
            st.session_state.slide_index = 0

with c2:

    try:
        st.markdown('<div class="slide-card">', unsafe_allow_html=True)

        st.image(
            slide_files[st.session_state.slide_index],
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    except:
        st.warning("Slides not found.")

# =========================================================
# LOAD DATASET
# =========================================================

try:
    df = pd.read_csv("final_atig_dataset.csv")

except Exception as e:

    st.error("Dataset file not found.")
    st.stop()

# =========================================================
# CLEAN COLUMNS
# =========================================================

df.columns = df.columns.str.strip()

# =========================================================
# FORCE COLUMN MAPPING
# =========================================================

flux_col = "fluxes"
current_col = "current_A"
voltage_col = "voltage_V"
speed_col = "travel_speed"
penetration_col = "penetration_mm"

# =========================================================
# CREATE MATERIAL COLUMN
# =========================================================

if "material" not in df.columns:
    df["material"] = "SS304"

material_col = "material"

# =========================================================
# VERIFY COLUMNS
# =========================================================

required_cols = [
    flux_col,
    current_col,
    voltage_col,
    speed_col,
    penetration_col,
    material_col
]

missing_cols = [c for c in required_cols if c not in df.columns]

if len(missing_cols) > 0:

    st.error(f"Missing columns: {missing_cols}")

    st.write("Available columns:")
    st.write(df.columns.tolist())

    st.stop()

# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna()

df[current_col] = pd.to_numeric(
    df[current_col],
    errors='coerce'
)

df[voltage_col] = pd.to_numeric(
    df[voltage_col],
    errors='coerce'
)

df[speed_col] = pd.to_numeric(
    df[speed_col],
    errors='coerce'
)

df[penetration_col] = pd.to_numeric(
    df[penetration_col],
    errors='coerce'
)

df = df.dropna()

# =========================================================
# ENCODERS
# =========================================================

material_encoder = LabelEncoder()
flux_encoder = LabelEncoder()

df["Material_encoded"] = material_encoder.fit_transform(
    df[material_col].astype(str)
)

df["Flux_encoded"] = flux_encoder.fit_transform(
    df[flux_col].astype(str)
)

# =========================================================
# FEATURES
# =========================================================

X = df[
    [
        current_col,
        voltage_col,
        speed_col,
        "Material_encoded",
        "Flux_encoded"
    ]
]

y = df[penetration_col]

# =========================================================
# RANDOM FOREST MODEL
# =========================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

# =========================================================
# PARAMETERS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 class="glow">
⚙️ Welding Parameters
</h1>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    current = st.slider(
        "Current (A)",
        80,
        180,
        120
    )

    voltage = st.slider(
        "Voltage (V)",
        8,
        20,
        12
    )

with col2:

    speed = st.slider(
        "Travel Speed (mm/min)",
        40,
        350,
        120
    )

    material = st.selectbox(
        "Material",
        sorted(df[material_col].astype(str).unique())
    )

flux = st.selectbox(
    "Flux Type",
    sorted(df[flux_col].astype(str).unique())
)

# =========================================================
# ENCODE INPUTS
# =========================================================

material_encoded = material_encoder.transform(
    [material]
)[0]

flux_encoded = flux_encoder.transform(
    [flux]
)[0]

# =========================================================
# PREDICTION INPUT
# =========================================================

input_df = pd.DataFrame({

    current_col:[current],
    voltage_col:[voltage],
    speed_col:[speed],
    "Material_encoded":[material_encoded],
    "Flux_encoded":[flux_encoded]

})

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(input_df)[0]

prediction = round(float(prediction), 2)

# =========================================================
# HEAT INPUT
# =========================================================

heat_input = round(
    ((voltage * current * 60)/(1000 * speed)),
    2
)

# =========================================================
# RELATIVE IMPROVEMENT
# =========================================================

try:

    base_df = df[
        df[flux_col].astype(str).str.lower().str.contains("no")
    ]

    if len(base_df) > 0:
        baseline = base_df[penetration_col].mean()

    else:
        baseline = prediction

    increase = (
        ((prediction - baseline)/baseline) * 100
    )

    increase = round(float(increase), 1)

except:
    increase = 0

# =========================================================
# RESULTS
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 class="glow">
📊 Prediction Results
</h1>
""", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
    <h3>Penetration Depth</h3>
    <h1>{prediction} mm</h1>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
    <h3>Relative Increase</h3>
    <h1>{increase}%</h1>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
    <h3>Heat Input</h3>
    <h1>{heat_input} kJ/mm</h1>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PENETRATION GRAPH
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 class="glow">
📈 AI Penetration Prediction Graph
</h1>
""", unsafe_allow_html=True)

currents = np.arange(80, 181, 5)

predictions = []

for c in currents:

    temp_df = pd.DataFrame({

        current_col:[c],
        voltage_col:[voltage],
        speed_col:[speed],
        "Material_encoded":[material_encoded],
        "Flux_encoded":[flux_encoded]

    })

    pred = model.predict(temp_df)[0]

    predictions.append(pred)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=currents,
        y=predictions,
        mode='lines+markers',
        name='Penetration Prediction'
    )
)

fig.update_layout(

    template="plotly_dark",

    title=f"Predicted Penetration vs Current ({flux})",

    xaxis_title="Current (A)",

    yaxis_title="Penetration Depth (mm)",

    height=500,

    paper_bgcolor="#0f172a",

    plot_bgcolor="#111827",

    font=dict(color="white")

)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 class="glow">
🧠 Random Forest Feature Importance
</h1>
""", unsafe_allow_html=True)

importance = model.feature_importances_

feature_names = [
    "Current",
    "Voltage",
    "Travel Speed",
    "Material",
    "Flux"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=importance_df["Feature"],
        y=importance_df["Importance"]
    )
)

fig2.update_layout(

    template="plotly_dark",

    title="Feature Importance",

    xaxis_title="Features",

    yaxis_title="Importance",

    height=450,

    paper_bgcolor="#0f172a",

    plot_bgcolor="#111827",

    font=dict(color="white")

)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# DATASET PREVIEW
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h1 class="glow">
📁 Dataset Preview
</h1>
""", unsafe_allow_html=True)

st.dataframe(df.head(20), use_container_width=True)

# =========================================================
# EXPLANATION
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style="
background:#0f172a;
padding:30px;
border-radius:25px;
border:1px solid #334155;
">

<h2 class="glow">How The AI Prediction Works</h2>

<p style="font-size:20px; line-height:2; color:#d1d5db;">

• The platform uses Random Forest Machine Learning.<br><br>

• It learns penetration behavior from literature-derived welding datasets.<br><br>

• Parameters include:
Current,
Voltage,
Travel Speed,
Material Type,
and Activated Flux Type.<br><br>

• Nano oxide fluxes such as TiO₂ and SiO₂ can improve penetration
through arc constriction and Marangoni convection effects.<br><br>

• The AI predicts penetration depth based on learned welding trends.

</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<center>

<a href="https://www.linkedin.com/in/harsha-varthanan/"
target="_blank"
style="
font-size:24px;
color:#38bdf8;
text-decoration:none;
font-weight:700;
">

Connect with me on LinkedIn

</a>

</center>
""", unsafe_allow_html=True)