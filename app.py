# =========================================================
# AI TIG / A-TIG WELD PREDICTION PLATFORM
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI TIG Weld Prediction",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp{
    background:#050816;
}

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

html, body, [class*="css"]{
    color:white;
}

/* TITLE */

.big-title{
    font-size:62px;
    font-weight:800;
    color:white;
    line-height:1.15;
}

.sub-title{
    font-size:30px;
    color:#7dd3fc;
    font-weight:600;
}

.description{
    font-size:21px;
    color:#d1d5db;
    line-height:2;
}

/* SECTION */

.section-title{
    font-size:42px;
    font-weight:800;
    color:#7dd3fc;
    margin-top:30px;
}

/* CARD */

.card{

    background:#111827;

    border-radius:25px;

    padding:30px;

    border:1px solid #1f2937;

    box-shadow:0px 0px 25px rgba(0,0,0,0.35);

    height:100%;
}

.card h2{
    color:#7dd3fc;
    font-size:30px;
}

.card ul{
    color:#f3f4f6;
    font-size:20px;
    line-height:2;
}

/* METRIC */

[data-testid="stMetric"]{

    background:#111827;

    border-radius:20px;

    padding:20px;

    border:1px solid #1f2937;
}

/* BUTTON */

.stButton>button{

    background:linear-gradient(
        135deg,
        #2563eb,
        #38bdf8
    );

    color:white;

    border:none;

    border-radius:16px;

    height:60px;

    font-size:22px;

    font-weight:700;

    box-shadow:0px 0px 20px rgba(56,189,248,0.4);
}

/* SELECT */

.stSelectbox > div > div{

    background:#111827;

    color:white;

    border-radius:12px;
}

/* MOBILE */

@media(max-width:768px){

    .big-title{
        font-size:38px;
    }

    .sub-title{
        font-size:20px;
    }

    .description{
        font-size:17px;
    }

    .section-title{
        font-size:30px;
    }

    .card ul{
        font-size:17px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

left_logo, center_logo, right_logo = st.columns([1,5,1])

with left_logo:

    if os.path.exists("assets/CEG_col1.png"):
        st.image("assets/CEG_col1.png", width=120)

with center_logo:

    st.markdown("""
    <div style="
        background:linear-gradient(
            135deg,
            #111827,
            #1e293b
        );



    <div class="big-title">
    AI-Based TIG / A-TIG<br>
    Weld Prediction
    </div>

    <br>

    <div class="sub-title">
    Nano Oxide Activated Fluxes for SS304 Welding
    </div>

    <br><br>

    <div class="description">
    AI-assisted weld penetration prediction platform integrating
    welding metallurgy, nano oxide activated fluxes,
    penetration trend visualization and intelligent
    machine learning analysis.
    </div>

    </div>
    """, unsafe_allow_html=True)

with right_logo:

    if os.path.exists("assets/iiti.png"):
        st.image("assets/iiti.png", width=120)

# =========================================================
# LITERATURE SLIDES
# =========================================================

st.markdown("""
<div class="section-title">
📘 Presentation Slides
""", unsafe_allow_html=True)

slide_images = []

for i in range(1,12):

    path = f"papers/slide{i}.png"

    if os.path.exists(path):
        slide_images.append(path)

if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

left_btn, center_img, right_btn = st.columns([1,8,1])

with left_btn:

    if st.button("⬅"):
        st.session_state.slide_index -= 1

with right_btn:

    if st.button("➡"):
        st.session_state.slide_index += 1

if len(slide_images) > 0:

    if st.session_state.slide_index < 0:
        st.session_state.slide_index = len(slide_images)-1

    if st.session_state.slide_index >= len(slide_images):
        st.session_state.slide_index = 0

    with center_img:

        st.markdown("""
        <div style="
            background:#111827;
            border-radius:25px;
            padding:12px;
            box-shadow:0px 0px 25px rgba(56,189,248,0.2);
        ">
        """, unsafe_allow_html=True)

        st.image(
            slide_images[st.session_state.slide_index],
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

else:

    st.warning("Add slide images inside papers folder")

# =========================================================
# OVERVIEW
# =========================================================

st.markdown("""
<div class="section-title">
⚡ TIG vs A-TIG Welding
</div>
""", unsafe_allow_html=True)

left,right = st.columns(2)

with left:

    st.markdown("""
    <div class="card">

    <h2>Conventional TIG Welding</h2>

    <ul>
    <li>Limited penetration depth</li>
    <li>Wide weld bead morphology</li>
    <li>Higher thermal distortion</li>
    <li>Larger heat affected zone</li>
    <li>Multiple welding passes required</li>

    </ul>

    </div>
    """, unsafe_allow_html=True)

with right:

    st.markdown("""
    <div class="card">

    <h2>A-TIG Welding</h2>

    <ul>

    <li>Reverse Marangoni convection</li>
    <li>Arc constriction mechanism</li>
    <li>Deeper weld penetration</li>
    <li>Narrow weld bead profile</li>
    <li>Improved welding efficiency</li>

    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PARAMETERS
# =========================================================

st.markdown("""
<div class="section-title">
⚙ Welding Parameters
</div>
""", unsafe_allow_html=True)

c1,c2 = st.columns(2)

with c1:

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

with c2:

    speed = st.slider(
        "Travel Speed (mm/min)",
        80,
        200,
        120
    )

    material = st.selectbox(
        "Material",
        [
            "SS304",
            "SS316",
            "Duplex 2205"
        ]
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
# RANDOM FOREST
# =========================================================

if os.path.exists("tig_model.pkl"):

    model = joblib.load("tig_model.pkl")
    encoder = joblib.load("flux_encoder.pkl")

else:

    dataset = pd.read_csv("final_atig_dataset.csv")

    encoder = LabelEncoder()

    dataset["flux_encoded"] = encoder.fit_transform(
        dataset["flux"]
    )

    X = dataset[
        [
            "current_A",
            "voltage_V",
            "travel_speed",
            "flux_encoded"
        ]
    ]

    y = dataset["penetration_mm"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X,y)

    joblib.dump(model,"tig_model.pkl")
    joblib.dump(encoder,"flux_encoder.pkl")

# =========================================================
# PREDICTION
# =========================================================

try:
    flux_encoded = encoder.transform([flux])[0]
except:
    flux_encoded = 0

input_df = pd.DataFrame({

    "current_A":[current],
    "voltage_V":[voltage],
    "travel_speed":[speed],
    "flux_encoded":[flux_encoded]
})

prediction = model.predict(input_df)[0]

# =========================================================
# METALLURGY FACTORS
# =========================================================

flux_factor = {

    "No Flux":1.0,
    "Fe2O3":1.15,
    "Al2O3":1.28,
    "Cr2O3":1.45,
    "TiO2":1.85,
    "SiO2":2.05,
    "TiO2+SiO2":2.15,
    "TiO2+Al2O3":1.95,
    "SiO2+Cr2O3":2.0
}

prediction *= flux_factor.get(flux,1)

prediction = max(1.5,prediction)
prediction = min(14,prediction)

heat_input = (
    voltage * current * 60
)/(1000*speed)

relative = ((prediction-3)/3)*100

# =========================================================
# RESULTS
# =========================================================

st.markdown("""
<div class="section-title">
📊 Prediction Results
</div>
""", unsafe_allow_html=True)

m1,m2,m3 = st.columns(3)

with m1:
    st.metric(
        "Penetration Depth",
        f"{prediction:.2f} mm"
    )

with m2:
    st.metric(
        "Relative Increase",
        f"{relative:.1f}%"
    )

with m3:
    st.metric(
        "Heat Input",
        f"{heat_input:.2f} kJ/mm"
    )

# =========================================================
# PENETRATION GRAPH
# =========================================================

st.markdown("""
<div class="section-title">
📈 Penetration Trend
</div>
""", unsafe_allow_html=True)

currents = np.arange(90,161,10)

values = []

for c in currents:

    val = (
        (c/100)
        * flux_factor.get(flux,1)
        * 1.8
    )

    values.append(val)

graph_df = pd.DataFrame({

    "Current":currents,
    "Penetration":values
})

fig = px.line(
    graph_df,
    x="Current",
    y="Penetration",
    markers=True
)

fig.update_traces(
    line=dict(
        color="#38bdf8",
        width=4
    )
)

fig.update_layout(

    paper_bgcolor="#111827",

    plot_bgcolor="#111827",

    font=dict(
        color="white",
        size=14
    ),

    height=350
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# WELD MORPHOLOGY
# =========================================================

st.markdown("""
<div class="section-title">
🧬 Predicted Weld Morphology
</div>
""", unsafe_allow_html=True)

fig2, ax = plt.subplots(figsize=(4,4))

fig2.patch.set_facecolor("#111827")
ax.set_facecolor("#111827")

ax.plot([-5,5],[0,0], color="white", linewidth=5)

depth = prediction / 2

if flux == "No Flux":
    width = 4.5

elif flux in ["Fe2O3","Al2O3"]:
    width = 3.0

elif flux == "Cr2O3":
    width = 2.4

else:
    width = 1.7

points = np.array([

    [-width,0],
    [-width/2,-depth*0.4],
    [0,-depth],
    [width/2,-depth*0.4],
    [width,0]
])

polygon = Polygon(
    points,
    closed=True,
    color="#38bdf8",
    alpha=0.85
)

ax.add_patch(polygon)

ax.set_xlim(-5,5)
ax.set_ylim(-8,2)

ax.grid(True,color="#374151")

ax.tick_params(colors="white")

ax.set_title(
    "Predicted Weld Cross Section",
    color="white",
    fontsize=15
)

st.pyplot(fig2)

# =========================================================
# LINKEDIN
# =========================================================

st.markdown("---")

st.markdown("""

<center>

<h2 style="
color:#7dd3fc;
font-size:34px;
">
Connect With Me
</h2>

<br>

<a href="https://www.linkedin.com/in/harsha-varthanan/"
target="_blank"
style="
font-size:24px;
color:#38bdf8;
text-decoration:none;
font-weight:700;
">

linkedin.com/in/harsha-varthanan/

</a>

<br><br>

<p style="
color:#9ca3af;
font-size:18px;
">

Developed by Harsha Varthanan

</p>

</center>

""", unsafe_allow_html=True)