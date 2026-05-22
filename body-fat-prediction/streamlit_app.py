import os
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Body Fat Prediction",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="auto",
)

LBS_TO_KG: float = 0.453592
IN_TO_M:   float = 0.0254
IN_TO_CM:  float = 2.54

LEAN_THRESHOLD:    float = 14.0
HEALTHY_THRESHOLD: float = 25.0

ACTIVITY_LEVELS: dict[str, float] = {
    "Sedentary (desk job)":              1.200,
    "Lightly active (1–3 days/week)":    1.375,
    "Moderately active (3–5 days/week)": 1.550,
    "Very active (6–7 days/week)":       1.725,
    "Extremely active (athlete)":        1.900,
}

MEASUREMENT_META: dict[str, dict] = {
    "Abdomen": {"ref": 83, "ideal": (70, 88)},
    "Chest":   {"ref": 95, "ideal": (90, 105)},
    "Hip":     {"ref": 97, "ideal": (85, 100)},
    "Thigh":   {"ref": 57, "ideal": (50, 60)},
    "Biceps":  {"ref": 31, "ideal": (28, 36)},
    "Forearm": {"ref": 28, "ideal": (24, 32)},
    "Neck":    {"ref": 37, "ideal": (34, 40)},
    "Knee":    {"ref": 38, "ideal": (35, 45)},
    "Ankle":   {"ref": 23, "ideal": (20, 26)},
    "Wrist":   {"ref": 18, "ideal": (16, 20)},
}

if "history"    not in st.session_state:
    st.session_state.history    = []
if "intro_done" not in st.session_state:
    st.session_state.intro_done = False


def inject_css() -> None:
    st.markdown("""
    <style>
    .stApp {
        background-image:
            linear-gradient(rgba(2,6,23,.88), rgba(15,23,42,.92)),
            url("https://4kwallpapers.com/images/wallpapers/bodybuilder-amoled-3840x2160-17344.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }

    [data-testid="stAppViewBlockContainer"],
    .stMainBlockContainer,
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
    }

    .hero-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
        padding: 0 20px;
        box-sizing: border-box;
    }

    .hero-section {
        position: relative; overflow: hidden;
        width: 100%;
        padding: 70px 50px;
        border-radius: 28px;
        background: linear-gradient(135deg,rgba(15,23,42,.92),rgba(30,41,59,.78));
        border: 2px solid rgba(96,165,250,.45);
        backdrop-filter: blur(18px);
        box-shadow: 0 15px 60px rgba(0,0,0,.55), 0 0 120px rgba(96,165,250,.12), inset 0 1px 0 rgba(255,255,255,.15);
        text-align: center;
    }
    .hero-section::before {
        content:""; position:absolute;
        width:350px; height:350px;
        background:rgba(96,165,250,.16);
        filter:blur(120px); top:-100px; left:-100px;
    }
    .hero-badge {
        display:inline-block; padding:8px 18px; border-radius:999px;
        background:rgba(99,102,241,.25); border:2px solid rgba(99,102,241,.65);
        color:#c4b5fd; font-size:14px; font-weight:600; margin-bottom:18px;
        box-shadow: 0 0 16px rgba(99,102,241,.3);
    }
    .hero-title {
        font-size: clamp(2rem,5vw,3.6rem); font-weight:900; line-height:1.1;
        background: linear-gradient(90deg,#60a5fa,#a78bfa,#22d3ee);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        margin-bottom:14px;
    }
    .hero-subtitle {
        font-size:1.1rem; color:#cbd5e1; max-width:720px; margin:auto; line-height:1.8;
    }

    [data-testid="stMetric"] {
        background: rgba(17,24,39,.75);
        border: 2px solid rgba(255,255,255,.28);
        padding: 18px; border-radius: 18px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,.45), inset 0 1px 0 rgba(255,255,255,.12);
        transition: .3s ease; position:relative; overflow:hidden;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        border-color: rgba(96,165,250,.7);
        box-shadow: 0 15px 40px rgba(59,130,246,.35), 0 0 0 1px rgba(96,165,250,.2);
    }
    [data-testid="stMetric"]::before {
        content:""; position:absolute; inset:0;
        background: linear-gradient(135deg,rgba(255,255,255,.08),transparent);
        pointer-events:none;
    }

    .result-banner {
        padding: 18px; border-radius: 16px;
        text-align: center; font-size: 20px; font-weight: 700;
        margin-bottom: 8px; color: white;
        animation: fadeIn .6s ease;
        border-width: 2px !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(15,23,42,.97);
        border-right: 3px solid rgba(96,165,250,.35);
    }

    .stButton > button {
        width:100%; border-radius:14px;
        background: linear-gradient(135deg,#4f46e5,#7c3aed);
        color:white; font-weight:bold; border:none;
        padding:.8rem 1rem; transition:.3s;
        box-shadow: 0 4px 18px rgba(79,70,229,.45);
    }
    .stButton > button:hover {
        transform:scale(1.03);
        box-shadow: 0 0 28px rgba(124,58,237,.6);
    }

    .stNumberInput input {
        background-color: rgba(255,255,255,.06) !important;
        color: white !important; border-radius: 12px !important;
        border: 2px solid rgba(255,255,255,.2) !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-size:15px; padding:10px 18px; border-radius:10px; color:white;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99,102,241,.3);
        box-shadow: 0 0 0 1px rgba(99,102,241,.5);
    }

    /* ── Plotly chart containers ── */
    [data-testid="stPlotlyChart"] > div {
        border: 2px solid rgba(255,255,255,.2);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 6px 24px rgba(0,0,0,.4);
    }

    /* ── st.info / st.success / st.warning ── */
    [data-testid="stAlert"] {
        border-width: 2px !important;
        border-left-width: 4px !important;
    }

    h2, h3, h4 { color: white !important; }

    ::-webkit-scrollbar { width:8px; }
    ::-webkit-scrollbar-thumb { background:#4f46e5; border-radius:8px; }

    @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
    </style>
    """, unsafe_allow_html=True)


HERO_HTML = """
<div class="hero-wrapper">
    <div class="hero-section">
        <div class="hero-badge">◆ ML-Powered Health Intelligence</div>
        <h1 class="hero-title">Body Fat Prediction System</h1>
        <p class="hero-subtitle">
            Advanced machine learning analysis for body composition,
            metabolic estimation, and personalised fitness insights.
        </p>
    </div>
</div>
"""


def show_intro_animation(hero_placeholder: st.delta_generator.DeltaGenerator) -> None:
    if not st.session_state.intro_done:
        st.session_state.intro_done = True
        components.html("""
        <style>
        .intro-overlay {
            position:fixed; top:0; left:0; width:100vw; height:100vh;
            background: transparent;
            display:flex; align-items:center; justify-content:center;
            flex-direction:column; z-index:999999;
            animation: fadeOut .8s ease 3s forwards;
        }
        .intro-title {
            font-size:36px; font-weight:800;
            background:linear-gradient(90deg,#60a5fa,#a78bfa,#22d3ee);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            animation:float 2s ease-in-out infinite;
        }
        .intro-sub { margin-top:10px; color:#cbd5e1; font-size:15px; }
        .loader {
            position: relative;
            margin-top: 25px;
            width: 90px; height: 90px;
            border-radius: 50%;
            background: rgba(15,23,42,0.6);
            border: 2px solid rgba(96,165,250,0.3);
            box-shadow: 0 0 30px rgba(96,165,250,0.2), inset 0 0 20px rgba(96,165,250,0.3);
            overflow: hidden;
            backdrop-filter: blur(8px);
        }
        .loader::before {
            content: '';
            position: absolute;
            top: 50%; left: 50%;
            width: 50%; height: 50%;
            background: conic-gradient(from 0deg, transparent 70%, rgba(96,165,250,0.9) 100%);
            transform-origin: 0 0;
            animation: radarScan 1.2s linear infinite;
        }
        .loader::after {
            content: '';
            position: absolute;
            inset: 4px;
            border-radius: 50%;
            border: 2px solid rgba(167,139,250,0.4);
        }
        @keyframes radarScan { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes float  { 50%{transform:translateY(-10px)} }
        @keyframes fadeOut{ to{opacity:0;visibility:hidden} }
        </style>
        <div class="intro-overlay">
            <div class="intro-title">◆ AI Body Fat System</div>
            <div class="intro-sub">Analyzing your body composition…</div>
            <div class="loader"></div>
        </div>
        """, height=700)
        time.sleep(3.8)
    hero_placeholder.markdown(HERO_HTML, unsafe_allow_html=True)


BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebook")


@st.cache_resource
def load_artifacts():
    paths = {
        "model":    os.path.join(MODEL_DIR,    "linear_regression_model.pkl"),
        "scaler":   os.path.join(MODEL_DIR,    "scaler.pkl"),
        "features": os.path.join(NOTEBOOK_DIR, "features.pkl"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        msg = (
            "Missing files:\n" + "\n".join(f"  • {p}" for p in missing)
            + f"\n\nProject root: `{BASE_DIR}`\n"
            "Run `streamlit run app.py` from the folder containing `models/` and `notebook/`."
        )
        return None, None, None, msg
    try:
        model    = joblib.load(paths["model"])
        scaler   = joblib.load(paths["scaler"])
        features = joblib.load(paths["features"])
        return model, scaler, features, None
    except Exception as exc:
        return None, None, None, str(exc)


model, scaler, features, load_error = load_artifacts()


def classify_body_fat(pct: float) -> tuple[str, str]:
    if pct < LEAN_THRESHOLD:
        return "LEAN",          "#22c55e"
    if pct < HEALTHY_THRESHOLD:
        return "HEALTHY",       "#3b82f6"
    return "HIGH BODY FAT",     "#ef4444"


def theme_color(pred: float | None = None) -> str:
    if pred is None:               return "#60a5fa"
    if pred >= HEALTHY_THRESHOLD:  return "#ef4444"
    if pred >= LEAN_THRESHOLD:     return "#3b82f6"
    return "#22c55e"


@st.cache_data
def compute_bmi(weight_lbs: float, height_in: float) -> float:
    h_m = height_in * IN_TO_M
    return (weight_lbs * LBS_TO_KG) / (h_m ** 2)


def classify_bmi(bmi: float) -> tuple[str, str]:
    if bmi < 18.5: return "Underweight", "#facc15"
    if bmi < 25.0: return "Normal",      "#22c55e"
    if bmi < 30.0: return "Overweight",  "#f97316"
    return "Obese", "#ef4444"


def compute_whr(waist: float, hip: float) -> float:
    return round(waist / hip, 3) if hip > 0 else 0.0


@st.cache_data
def compute_bmr(weight_lbs: float, height_in: float, age: int, sex: str = "Male") -> float:
    w_kg = weight_lbs * LBS_TO_KG
    h_cm = height_in  * IN_TO_CM
    if sex == "Male":
        return 88.362 + (13.397 * w_kg) + (4.799 * h_cm) - (5.677 * age)
    return 447.593 + (9.247 * w_kg) + (3.098 * h_cm) - (4.330 * age)


def tdee_from_activity(bmr: float, level: str) -> float:
    multiplier = ACTIVITY_LEVELS.get(level, 1.375)
    return bmr * multiplier


def body_fat_percentile(pct: float) -> int:
    breakpoints = [
        (5,1),(10,5),(14,15),(18,35),(22,55),
        (25,65),(30,80),(35,90),(40,96),(50,99),
    ]
    for threshold, pctile in breakpoints:
        if pct <= threshold:
            return pctile
    return 99


def weeks_to_goal(current_bf: float, goal_bf: float,
                  weight_lbs: float, tdee: float) -> float | None:
    if abs(current_bf - goal_bf) < 0.5:
        return 0.0
    fat_lbs_now  = weight_lbs * (current_bf / 100)
    fat_lbs_goal = weight_lbs * (goal_bf   / 100)
    delta_lbs    = fat_lbs_now - fat_lbs_goal
    weekly_rate  = (500 * 7) / 3500
    weeks        = abs(delta_lbs) / weekly_rate
    return round(weeks, 1)


def build_record(prediction: float, bmi: float, whr: float,
                 bmr: float, tdee: float, health_score: float,
                 category: str, input_dict: dict) -> dict:
    return {
        "Timestamp":    pd.Timestamp.now().strftime("%H:%M:%S"),
        "Body Fat %":   round(prediction, 2),
        "BMI":          round(bmi, 2),
        "WHR":          whr,
        "BMR (kcal)":   round(bmr),
        "TDEE (kcal)":  round(tdee),
        "Health Score": round(health_score),
        "Category":     category,
        **input_dict,
    }


inject_css()

# Interactive 3D Background (Vanta.NET)
components.html("""
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
<div id="vanta-bg" style="width:100vw;height:100vh;position:absolute;top:0;left:0;"></div>
<script>
  VANTA.NET({
    el: "#vanta-bg",
    mouseControls: true,
    touchControls: true,
    gyroControls: false,
    minHeight: 200.00,
    minWidth: 200.00,
    scale: 1.00,
    scaleMobile: 1.00,
    color: 0x60a5fa,
    backgroundColor: 0x0f172a,
    points: 12.00,
    maxDistance: 22.00,
    spacing: 16.00
  })

  // Hack to make iframe fullscreen background
  const frame = window.frameElement;
  if (frame) {
      frame.style.position = 'fixed';
      frame.style.top = '0';
      frame.style.left = '0';
      frame.style.width = '100vw';
      frame.style.height = '100vh';
      frame.style.zIndex = '-1';
      frame.style.border = 'none';
  }
  const parentDoc = window.parent.document;
  const stApp = parentDoc.querySelector('.stApp');
  const stHeader = parentDoc.querySelector('[data-testid="stHeader"]');
  if(stApp) {
      stApp.style.background = 'transparent';
  }
  if(stHeader) {
      stHeader.style.background = 'transparent';
  }
</script>
""", height=0)

hero_placeholder = st.empty()
show_intro_animation(hero_placeholder)

if load_error:
    st.error(
        f"△ Could not load model files: {load_error}\n\n"
        "Make sure `models/` and `notebook/` folders exist alongside `app.py`."
    )

with st.sidebar.expander("How to Measure Correctly", expanded=False):
    st.markdown("""
    <style>
    .section-box {
        padding: 10px 12px; margin-bottom: 16px; border-radius: 12px;
        background: rgba(255,255,255,0.06);
        border: 2px solid rgba(255,255,255,0.3);
    }
    .upper { border-left: 5px solid #60a5fa; margin-top: -14px; }
    .core  { border-left: 5px solid #a78bfa; }
    .lower { border-left: 5px solid #34d399; }
    .small { border-left: 5px solid #fbbf24; }
    .title { font-weight: 700; font-size: 14px; margin-bottom: 6px; color: white; }
    .text  { font-size: 13px; color: #cbd5e1; line-height: 1.5; }
    .warning {
        padding: 10px; border-radius: 12px;
        background: rgba(239,68,68,0.12); border: 2px solid rgba(239,68,68,0.55);
        color: #fecaca; font-size: 12.5px; margin-top: 14px; margin-bottom: 16px;
    }
    .tip {
        padding: 10px; border-radius: 12px;
        background: rgba(59,130,246,0.12); border: 2px solid rgba(59,130,246,0.55);
        color: #bfdbfe; font-size: 12.5px; margin-top: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-box upper">
        <div class="title">Upper Body</div>
        <div class="text">
        <b>Neck</b><br>↳ Just below the larynx, tape relaxed<br><br>
        <b>Chest</b><br>↳ At nipple level, arms relaxed, normal breathing<br><br>
        <b>Biceps</b><br>↳ Midpoint of upper arm (keep flexed or relaxed consistently)<br><br>
        <b>Forearm</b><br>↳ Widest point, hand open and relaxed
        </div>
    </div>
    <div class="section-box core">
        <div class="title">Core / Torso</div>
        <div class="text">
        <b>Abdomen</b><br>↳ At navel level, relaxed stomach (do not suck in)<br><br>
        <b>Hip</b><br>↳ Widest part of buttocks, feet together
        </div>
    </div>
    <div class="section-box lower">
        <div class="title">Lower Body</div>
        <div class="text">
        <b>Thigh</b><br>↳ Upper thigh just below glutes<br><br>
        <b>Knee</b><br>↳ Around kneecap (natural standing position)<br><br>
        <b>Ankle</b><br>↳ Narrowest point above ankle bone
        </div>
    </div>
    <div class="section-box small">
        <div class="title">Small Joints</div>
        <div class="text">
        <b>Wrist</b><br>↳ Just below wrist bone, relaxed hand
        </div>
    </div>
    <div class="tip"><b>Note :</b> Consistency matters more than perfection. Track changes over time, not single measurements.</div>
    <div class="warning"><b>Note :</b> Measure at the same time each day (preferably morning), and avoid measuring after meals or workouts.</div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("## 👤 Your Profile")

st.sidebar.markdown("### 📋 Basic Info")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.number_input("Age", 1, 100, 25)
st.sidebar.caption("Used for metabolic rate & prediction adjustment")

st.sidebar.divider()

st.sidebar.markdown("### ⚡ Lifestyle & Goals")
activity = st.sidebar.selectbox("Daily Activity", list(ACTIVITY_LEVELS.keys()), index=1)
st.sidebar.caption("Affects calorie burn (TDEE estimation)")
goal_bf = st.sidebar.slider("Target Body Fat %", 5, 40, 18,
                             help="We'll estimate how many weeks to reach this goal.")

st.sidebar.divider()

st.sidebar.markdown("### ⚖️ Body Stats")
col1, col2 = st.sidebar.columns(2)
with col1:
    weight = st.number_input("Weight (lbs)", 50, 400, 170)
with col2:
    height = st.number_input("Height (in)",  40,  90,  68)

bmi_preview = compute_bmi(weight, height)
st.sidebar.info(f"Estimated BMI: **{bmi_preview:.1f}**")

st.sidebar.divider()

st.sidebar.markdown("### 📏 Measurements (cm)")
st.sidebar.caption("Tip: consistency matters more than precision")

with st.sidebar.expander("Upper Body", expanded=True):
    neck    = st.slider("Neck",    20,  60,  38)
    chest   = st.slider("Chest",   50, 180, 100)
    biceps  = st.slider("Biceps",  15,  60,  32)
    forearm = st.slider("Forearm", 15,  50,  28)
    wrist   = st.slider("Wrist",   10,  30,  18)

with st.sidebar.expander("Core", expanded=True):
    abdomen = st.slider("Abdomen", 50, 180,  90)
    hip     = st.slider("Hip",     50, 180,  95)

with st.sidebar.expander("Lower Body", expanded=True):
    thigh   = st.slider("Thigh",   20, 100,  55)
    knee    = st.slider("Knee",    20,  70,  40)
    ankle   = st.slider("Ankle",   10,  40,  22)

_, col_btn, _ = st.sidebar.columns([1, 30, 1])
with col_btn:
    predict_clicked = st.button(
        "Predict Now",
        disabled=(model is None),
        use_container_width=True,
    )

if not predict_clicked and not st.session_state.history:
    st.markdown("""
    <style>
    section[data-testid="stMain"] { overflow: hidden !important; }
    [data-testid="stAppViewBlockContainer"],
    .stMainBlockContainer,
    .block-container { padding-bottom: 0 !important; }
    </style>
    """, unsafe_allow_html=True)
else:
    transition_css = "transition: all 0.5s ease-in-out;" if (predict_clicked and not st.session_state.history) else ""
    st.markdown(f"""
<style>
.hero-wrapper {{
    height: auto !important;
    padding-top: 30px !important;
    padding-bottom: 20px !important;
    {transition_css}
}}
</style>
""", unsafe_allow_html=True)

if predict_clicked:

    input_dict: dict = {
        "Age": age, "Weight": weight, "Height": height,
        "Neck": neck, "Chest": chest, "Abdomen": abdomen,
        "Hip": hip, "Thigh": thigh, "Knee": knee,
        "Ankle": ankle, "Biceps": biceps, "Forearm": forearm,
        "Wrist": wrist,
    }

    try:
        input_df     = pd.DataFrame([input_dict])[features]
        input_scaled = scaler.transform(input_df)
    except KeyError as exc:
        st.error(f"× Feature mismatch: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"× Preprocessing error: {exc}")
        st.stop()

    with st.spinner("Analysing body composition…"):
        time.sleep(1.2)

    prediction   = float(np.clip(model.predict(input_scaled)[0], 0, 60))
    color        = theme_color(prediction)
    category, category_color = classify_body_fat(prediction)

    LIVE_STEPS = [
        (0,  "·  Reading measurements…"),
        (20, "·  Scaling features…"),
        (40, "·  Running inference…"),
        (65, "·  Computing derived metrics…"),
        (85, "·  Finalising report…"),
    ]

    live_text = st.empty()
    bar        = st.progress(0)
    step_iter  = iter(LIVE_STEPS)
    next_threshold, next_msg = next(step_iter)

    for i in range(100):
        if i >= next_threshold:
            live_text.markdown(
                f"""<div style="text-align:center; font-size:15px; color:#94a3b8;
                    letter-spacing:.4px; padding:6px 0; animation: fadeIn .4s ease;">
                    {next_msg}</div>""",
                unsafe_allow_html=True,
            )
            try:
                next_threshold, next_msg = next(step_iter)
            except StopIteration:
                next_threshold = 101
        time.sleep(0.018)
        bar.progress(i + 1)

    live_text.empty()
    bar.empty()

    st.markdown(f"""
    <div class="result-banner"
         style="background:linear-gradient(135deg,{color}44,{color}11);
                border:3px solid {color};
                box-shadow: 0 0 24px {color}55, inset 0 1px 0 rgba(255,255,255,.15);">
        Analysis Complete
    </div>
    """, unsafe_allow_html=True)

    bmi                  = compute_bmi(weight, height)
    bmi_label, _         = classify_bmi(bmi)
    whr                  = compute_whr(abdomen, hip)
    bmr                  = compute_bmr(weight, height, age, sex)
    tdee                 = tdee_from_activity(bmr, activity)
    health_score         = max(0.0, 100 - prediction * 2)
    percentile           = body_fat_percentile(prediction)
    fat_mass_lbs         = round(weight * prediction / 100, 1)
    lean_mass_lbs        = round(weight - fat_mass_lbs, 1)
    weeks                = weeks_to_goal(prediction, goal_bf, weight, tdee)

    record = build_record(prediction, bmi, whr, bmr, tdee,
                          health_score, category, input_dict)
    st.session_state.history.append(record)

    st.markdown(f"""
    <div style="padding:18px;border-radius:16px;
                background:linear-gradient(135deg,{category_color}cc,{category_color}88);
                border:3px solid {category_color};
                box-shadow:0 6px 28px {category_color}55;
                font-size:22px;font-weight:bold;text-align:center;
                margin-bottom:18px;color:white;letter-spacing:1px;">
        {category}
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Body Fat %",   f"{prediction:.1f}%",
                  f"{prediction - 20:+.1f}% vs avg", delta_color="inverse")
    with c2:
        st.metric("BMI",          f"{bmi:.1f}", bmi_label)
    with c3:
        st.metric("WHR",          f"{whr:.2f}",
                  "High risk" if whr > 0.90 else "Normal", delta_color="inverse")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Fat Mass",     f"{fat_mass_lbs} lbs")
    with c5:
        st.metric("Lean Mass",    f"{lean_mass_lbs} lbs")
    with c6:
        st.metric("Health Score", f"{health_score:.0f}/100")

    st.progress(min(int(prediction * 100 / 60), 100))

    if weeks == 0.0:
        st.success("✓  You're already at your goal body fat percentage.")
    elif prediction > goal_bf:
        st.info(
            f"To reach **{goal_bf}% body fat** (~500 kcal/day deficit), "
            f"estimated **{weeks} weeks** at your current weight."
        )
    else:
        st.info(
            f"To reach **{goal_bf}% body fat** (~250 kcal/day surplus), "
            f"estimated **{weeks} weeks** at your current weight."
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Predictions", "Health", "Impact", "Recommendations", "Nutrition", "History",
    ])

    # ── TAB 1 ──────────────────────────────────────────────────────────
    with tab1:

        # ─────────────────────────────────────────────
        # GAUGE SECTION
        # ─────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Body Fat Gauge")

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Body Fat Percentage"},
                number={
                    "suffix": "%",
                    "font": {"size": 38, "color": "white"},
                },
                gauge={
                    "axis": {
                        "range": [0, 60],
                        "tickcolor": "white",
                        "tickfont": {"color": "white"},
                    },
                    "bar": {"color": category_color, "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,0.06)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.25)",
                    "steps": [
                        {"range": [0, 14], "color": "#14532d"},
                        {"range": [14, 25], "color": "#1e3a8a"},
                        {"range": [25, 60], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": prediction,
                    },
                },
            ))

            gauge_fig.update_layout(
                paper_bgcolor="#0b0f1a",
                plot_bgcolor="#0b0f1a",
                font_color="white",
                height=360,
                margin=dict(t=55, b=20, l=25, r=25),
            )

            st.plotly_chart(gauge_fig, use_container_width=True)

            st.markdown("""
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:-8px;margin-bottom:10px;">
                <span style="color:#86efac;font-size:12.5px;">● Lean: &lt;14%</span>
                <span style="color:#93c5fd;font-size:12.5px;">● Healthy: 14–25%</span>
                <span style="color:#fca5a5;font-size:12.5px;">● High: &gt;25%</span>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.subheader("BMI Gauge")

            bmi_label, bmi_color = classify_bmi(bmi)

            bmi_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(bmi, 1),
                title={"text": "Body Mass Index"},
                number={
                    "font": {"size": 38, "color": "white"},
                },
                gauge={
                    "axis": {
                        "range": [10, 50],
                        "tickcolor": "white",
                        "tickfont": {"color": "white"},
                    },
                    "bar": {"color": bmi_color, "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,0.06)",
                    "borderwidth": 1,
                    "bordercolor": "rgba(255,255,255,0.25)",
                    "steps": [
                        {"range": [10, 18.5], "color": "#713f12"},
                        {"range": [18.5, 25], "color": "#14532d"},
                        {"range": [25, 30], "color": "#78350f"},
                        {"range": [30, 50], "color": "#7f1d1d"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.75,
                        "value": bmi,
                    },
                },
            ))

            bmi_gauge.update_layout(
                paper_bgcolor="#0b0f1a",
                plot_bgcolor="#0b0f1a",
                font_color="white",
                height=360,
                margin=dict(t=55, b=20, l=25, r=25),
            )

            st.plotly_chart(bmi_gauge, use_container_width=True)

            st.markdown("""
            <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:-8px;margin-bottom:10px;">
                <span style="color:#facc15;font-size:12.5px;">● Underweight: &lt;18.5</span>
                <span style="color:#86efac;font-size:12.5px;">● Normal: 18.5–24.9</span>
                <span style="color:#fdba74;font-size:12.5px;">● Overweight: 25–29.9</span>
                <span style="color:#fca5a5;font-size:12.5px;">● Obese: ≥30</span>
            </div>
            """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────
        # RADAR CHART
        # ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Measurements vs Healthy Reference")

        radar_keys = ["Abdomen", "Chest", "Hip", "Thigh", "Biceps", "Forearm"]

        user_vals = [input_dict.get(k, 0) for k in radar_keys]
        ref_vals = [MEASUREMENT_META[k]["ref"] for k in radar_keys]

        radar_fig = go.Figure()

        radar_fig.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=radar_keys + [radar_keys[0]],
            fill="toself",
            name="Your Measurements",
            line_color="#60a5fa",
            line_width=3,
            fillcolor="rgba(96,165,250,0.24)",
        ))

        radar_fig.add_trace(go.Scatterpolar(
            r=ref_vals + [ref_vals[0]],
            theta=radar_keys + [radar_keys[0]],
            fill="toself",
            name="Healthy Reference",
            line_color="#22c55e",
            line_width=2,
            fillcolor="rgba(34,197,94,0.13)",
            opacity=0.75,
        ))

        radar_fig.update_layout(
            polar=dict(
                bgcolor="#0b0f1a",
                radialaxis=dict(
                    visible=True,
                    gridcolor="rgba(255,255,255,.25)",
                    linecolor="rgba(255,255,255,.35)",
                    tickfont=dict(color="white"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,.20)",
                    linecolor="rgba(255,255,255,.35)",
                    tickfont=dict(color="white", size=12),
                ),
            ),
            paper_bgcolor="#0b0f1a",
            plot_bgcolor="#0b0f1a",
            font_color="white",
            height=480,
            legend=dict(
                font=dict(color="white"),
                orientation="h",
                yanchor="bottom",
                y=-0.18,
                xanchor="center",
                x=0.5,
            ),
            margin=dict(t=45, b=70, l=40, r=40),
        )

        st.plotly_chart(radar_fig, use_container_width=True)

        # ─────────────────────────────────────────────
        # EXPORT SECTION
        # ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Export Result")

        csv_bytes = pd.DataFrame([record]).to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Result as CSV",
            data=csv_bytes,
            file_name="body_fat_result.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── TAB 2 ──────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Health Insights")
        # Determine sex-specific thresholds
        is_male = (sex == "Male")
        whr_high = 0.90 if is_male else 0.85
        whr_mod  = 0.85 if is_male else 0.80
        abd_high = 102  if is_male else 88

        whr_risk = (
            "▲ High Risk"     if whr > whr_high else
            "△ Moderate Risk" if whr > whr_mod else
            "○ Low Risk"
        )
        st.info(f"**Waist-to-Hip Ratio:** {whr} — {whr_risk}")

        flags = []
        if abdomen > abd_high: flags.append(("ti-alert-circle", "#ef4444", "High Abdomen Circumference", f"Your abdomen ({abdomen} cm) is above the {abd_high} cm threshold, a strong predictor of visceral fat."))
        if weight > 180:       flags.append(("ti-scale", "#f97316", "Higher Weight Profile", "Weight contributes significantly to the overall fat mass estimate."))
        if age > 40:           flags.append(("ti-calendar-stats", "#f59e0b", "Age Factor", "Metabolism naturally slows past 40. Resistance training is key to maintaining muscle."))
        if thigh > 60:         flags.append(("ti-ruler-measure", "#3b82f6", "Large Thigh Size", "A larger thigh measurement has a moderate upward influence on the prediction."))
        if whr > whr_high:     flags.append(("ti-heart-broken", "#ef4444", "Elevated WHR", f"WHR > {whr_high} indicates elevated cardiovascular risk for {sex.lower()}s."))
        elif whr > whr_mod:    flags.append(("ti-heart-rate-monitor", "#f97316", "Moderate WHR", f"WHR > {whr_mod} indicates moderate cardiovascular risk."))
        if bmi > 30:           flags.append(("ti-activity", "#ef4444", "BMI in Obese Range", "Consider consulting a healthcare provider for a clinical assessment."))

        if not flags:
            flags.append(("ti-check", "#22c55e", "Measurements Balanced", "Your measurements appear balanced with no major risk outliers — great job!"))

        cards_html = ""
        for icon, color, title, desc in flags:
            cards_html += f'''
            <div style="display:flex; gap:14px; background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.15); 
                        border-left:4px solid {color}; border-radius:12px; padding:16px; margin-bottom:12px; box-shadow:0 4px 12px rgba(0,0,0,0.2);">
                <div style="display:flex; align-items:center; justify-content:center; width:38px; height:38px; border-radius:8px; background:{color}22; color:{color}; flex-shrink:0;">
                    <i class="ti {icon}" style="font-size:20px;"></i>
                </div>
                <div>
                    <div style="color:#f1f5f9; font-weight:600; font-size:15px; margin-bottom:4px;">{title}</div>
                    <div style="color:#94a3b8; font-size:13.5px; line-height:1.5;">{desc}</div>
                </div>
            </div>'''

        components.html(f'''
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
        <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: transparent; margin: 0; padding: 4px; }}
        ::-webkit-scrollbar {{ width:8px; }}
        ::-webkit-scrollbar-thumb {{ background:#4f46e5; border-radius:8px; }}
        </style>
        {cards_html}
        ''', height=min(len(flags) * 90 + 20, 360), scrolling=True)

        st.info(f"Body composition classified as **{category}** (estimated top {100 - percentile}% of population).")

        st.subheader("Position in ACE Range")
        range_fig = go.Figure()
        
        if is_male:
            brackets = [
                ("Essential", 0, 5, "#3b82f6"), ("Athletes", 5, 13, "#10b981"), 
                ("Fitness", 13, 17, "#22c55e"), ("Average", 17, 24, "#eab308"), ("Obese", 24, 60, "#ef4444")
            ]
        else:
            brackets = [
                ("Essential", 0, 13, "#3b82f6"), ("Athletes", 13, 20, "#10b981"), 
                ("Fitness", 20, 24, "#22c55e"), ("Average", 24, 31, "#eab308"), ("Obese", 31, 60, "#ef4444")
            ]

        for label, x0, x1, col in brackets:
            range_fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                                fillcolor=col, opacity=0.35, line_width=0)
            range_fig.add_annotation(x=(x0 + x1) / 2, y=0.5, text=label,
                                     showarrow=False, font=dict(color="white", size=11))
            
        range_fig.add_shape(type="line", x0=prediction, x1=prediction, y0=0, y1=1,
                            line=dict(color="white", width=4))
        range_fig.add_annotation(x=prediction, y=1.08, text=f"You: {prediction:.1f}%",
                                 showarrow=False, font=dict(color="white", size=13, weight="bold"))
        
        range_fig.update_layout(
            height=200, paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
            margin=dict(t=40, b=30, l=20, r=20),
            xaxis=dict(title=f"Body Fat % (ACE Standard for {sex}s)", range=[0, 60], tickfont=dict(color="white"),
                       gridcolor="rgba(255,255,255,.25)", linecolor="rgba(255,255,255,.35)"),
            yaxis=dict(visible=False, range=[0, 1.2]),
        )
        st.plotly_chart(range_fig, use_container_width=True)

        components.html("""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
        <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: transparent; margin: 0; padding: 4px; color:#e2e8f0; }
        ::-webkit-scrollbar { width:8px; }
        ::-webkit-scrollbar-thumb { background:#4f46e5; border-radius:8px; }
        .model-card {
            background: rgba(17,24,39,0.7);
            border: 1px solid rgba(255,255,255,0.15);
            border-left: 4px solid #8b5cf6;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            margin-top: 10px;
        }
        .header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
        .icon { display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 8px; background: rgba(139,92,246,0.2); color: #c4b5fd; font-size: 20px; }
        .title { font-weight: 600; font-size: 16px; color: #f1f5f9; }
        .content { font-size: 13.5px; color: #94a3b8; line-height: 1.6; margin-bottom: 16px; }
        .highlight { color: #c4b5fd; font-weight: 500; }
        .disclaimer { 
            background: rgba(239,68,68,0.1); 
            border-left: 3px solid rgba(239,68,68,0.5); 
            padding: 10px 14px; 
            border-radius: 6px; 
            font-size: 12.5px; 
            color: #fca5a5; 
            display: flex; gap: 10px; align-items: flex-start;
        }
        </style>
        <div class="model-card">
            <div class="header">
                <div class="icon"><i class="ti ti-brain"></i></div>
                <div class="title">About the AI Model</div>
            </div>
            <div class="content">
                This system uses a <span class="highlight">Linear Regression</span> algorithm trained on a comprehensive dataset of anthropometric body measurements (circumferences, weight, and height). Features are dynamically scaled before prediction. To ensure physiologically realistic results, the final body fat percentage output is clipped to a range of 0% to 60%.
            </div>
            <div class="disclaimer">
                <i class="ti ti-alert-triangle" style="margin-top:2px;"></i>
                <div><b>Disclaimer:</b> This tool is designed for educational and informational tracking purposes only. It is not a substitute for clinical assessment. Always consult a healthcare professional for medical advice.</div>
            </div>
        </div>
        """, height=220, scrolling=True)

    # ── TAB 3 ──────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Key Drivers")
        coeff_path = os.path.join(MODEL_DIR, "coefficients.csv")

        if not os.path.exists(coeff_path):
            st.warning(
                "△  models/coefficients.csv not found. "
                "Generate it from your training notebook and place it in `models/`.\n\n"
                "The ideal-range section below will not render until that file is present."
            )
        else:
            coeff_df = pd.read_csv(coeff_path)
            coeff_df["Percentage"] = (coeff_df["Abs"] / coeff_df["Abs"].sum()) * 100

            user_values = {
                "Neck": neck, "Chest": chest, "Abdomen": abdomen, "Hip": hip,
                "Thigh": thigh, "Knee": knee, "Ankle": ankle,
                "Biceps": biceps, "Forearm": forearm, "Wrist": wrist,
            }

            html_cards = ""
            for feature, meta in MEASUREMENT_META.items():
                if feature not in user_values:
                    continue
                low, high  = meta["ideal"]
                user_val   = user_values[feature]
                
                if user_val < low:
                    status = "Below Ideal"
                    color  = "#3b82f6"
                    icon   = "ti-arrow-down"
                    pct    = max(20, (user_val / low) * 100)
                elif user_val > high:
                    status = "Above Ideal"
                    color  = "#ef4444"
                    icon   = "ti-arrow-up"
                    pct    = min(100, (user_val / high) * 100)
                else:
                    status = "Within Range"
                    color  = "#22c55e"
                    icon   = "ti-check"
                    pct    = 100
                    
                html_cards += f'''
                <div class="m-card">
                    <div class="m-header">
                        <div class="m-title">{feature}</div>
                        <div class="m-status" style="color:{color}; background:{color}22;">
                            <i class="ti {icon}"></i> {status}
                        </div>
                    </div>
                    <div class="m-values">
                        <div class="m-user">{user_val} <span class="m-unit">cm</span></div>
                        <div class="m-ideal">Ideal: {low}–{high}</div>
                    </div>
                    <div class="m-bar-bg">
                        <div class="m-bar-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                </div>
                '''
            
            components.html(f'''
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
            <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: transparent; margin: 0; padding: 4px; color:#e2e8f0; }}
            ::-webkit-scrollbar {{ width:8px; }}
            ::-webkit-scrollbar-thumb {{ background:#4f46e5; border-radius:8px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 16px; margin-bottom: 20px; }}
            .m-card {{
                background: rgba(17,24,39,0.7);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 14px; padding: 16px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                transition: transform 0.2s, border-color 0.2s;
            }}
            .m-card:hover {{ transform: translateY(-3px); border-color: rgba(96,165,250,0.5); box-shadow: 0 8px 20px rgba(96,165,250,0.15); }}
            .m-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
            .m-title {{ font-weight: 600; font-size: 15px; color: #f1f5f9; letter-spacing: 0.3px; }}
            .m-status {{ font-size: 11px; font-weight: 700; padding: 4px 8px; border-radius: 6px; display: flex; align-items: center; gap: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
            .m-values {{ display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 12px; }}
            .m-user {{ font-size: 24px; font-weight: 700; color: white; line-height: 1; }}
            .m-unit {{ font-size: 12px; color: #94a3b8; font-weight: 600; margin-left: 2px; }}
            .m-ideal {{ font-size: 12.5px; color: #94a3b8; font-weight: 500; }}
            .m-bar-bg {{ height: 6px; background: rgba(255,255,255,0.1); border-radius: 999px; overflow: hidden; }}
            .m-bar-fill {{ height: 100%; border-radius: 999px; transition: width 0.8s ease-out; }}
            </style>
            <div class="grid">
                {html_cards}
            </div>
            ''', height=360, scrolling=True)

            table_df = (
                coeff_df.sort_values("Percentage", ascending=False)
                        .reset_index(drop=True)
            )
            table_df["Rank"]         = range(1, len(table_df) + 1)
            table_df["Feature"]      = table_df["Feature"].str.replace("_", " ")
            table_df["Percentage"]   = table_df["Percentage"].round(2)
            table_df["Impact Level"] = table_df["Percentage"].apply(
                lambda x: "▲  High" if x > 10 else "◇  Medium" if x > 5 else "○  Low"
            )

            st.markdown("<h3 style='text-align:center;'>Contribution Table</h3>",
                        unsafe_allow_html=True)

            html_rows = "".join(
                f"<tr>"
                f"<td style='color:#a78bfa;font-weight:bold'>{row['Rank']}</td>"
                f"<td>{row['Feature']}</td>"
                f"<td>{row['Percentage']}%</td>"
                f"<td style='color:{'#ef4444' if '▲' in row['Impact Level'] else '#3b82f6' if '◇' in row['Impact Level'] else '#22c55e'};font-weight:bold'>{row['Impact Level']}</td>"
                f"</tr>"
                for _, row in table_df.iterrows()
            )
            components.html(f"""
            <style>
            .ct{{width:95%;border-collapse:collapse;background:rgba(17,24,39,.9);
                border-radius:18px;overflow:hidden;color:white;text-align:center;
                box-shadow:0 10px 40px rgba(0,0,0,.5);font-family:Arial;margin:auto;
                border:2px solid rgba(255,255,255,.3);}}
            .ct th{{padding:16px;background:linear-gradient(90deg,#4f46e5,#7c3aed);
                    border-bottom:2px solid rgba(255,255,255,.3);}}
            .ct td{{padding:12px;border-bottom:1px solid rgba(255,255,255,.2);}}
            .ct tr:nth-child(even){{background:rgba(255,255,255,.05)}}
            .ct tr:hover{{background:rgba(99,102,241,.25)}}
            </style>
            <table class="ct">
            <tr><th>Rank</th><th>Feature</th><th>Contribution %</th><th>Impact</th></tr>
            {html_rows}
            </table>
            """, height=min(len(table_df) * 44 + 80, 700), scrolling=True)

    # ── TAB 4 ──────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Recommendations")

        if prediction >= HEALTHY_THRESHOLD:
            status_html = """
            <div style="display:flex;align-items:center;gap:12px;
                        background:rgba(239,68,68,0.14);
                        border:2px solid rgba(239,68,68,0.7);
                        border-left:5px solid #ef4444;
                        box-shadow:0 4px 20px rgba(239,68,68,0.25);
                        border-radius:12px;padding:14px 18px;margin-bottom:24px;">
                <span style="font-size:22px;color:#fca5a5;">↑</span>
                <div>
                    <div style="font-weight:700;color:#fca5a5;font-size:15px;">Body Fat Above Healthy Range</div>
                    <div style="color:#94a3b8;font-size:13px;margin-top:2px;">
                        Focus on a steady deficit — not an aggressive cut. Slow progress sticks.
                    </div>
                </div>
            </div>
            """
            recs = [
                ("01", "Cardio", "running · cycling · rowing",
                 "3–4 sessions per week, 30–45 min each. Zone 2 pace — you should be able to hold a conversation. Skip HIIT until you have a base; it drives hunger up more than it burns fat."),
                ("02", "Calorie Deficit", "nutrition · intake",
                 "Aim for 300–500 kcal below your TDEE. That puts you at 0.5–1 lb loss per week — fast enough to see progress, slow enough to keep muscle. Track for two weeks before adjusting."),
                ("03", "Resistance Training", "weights · muscle",
                 "Lift 2–3× per week. Compound movements — squat, hinge, press, pull. You don't need to go heavy; just go consistently. Muscle kept during a cut raises your floor metabolism."),
                ("04", "Protein Intake", "diet · recovery",
                 f"Hit {round(weight * LBS_TO_KG * 1.8)}–{round(weight * LBS_TO_KG * 2.2)} g of protein per day. Spread it across meals. It's the single most important lever for holding onto muscle while in a deficit."),
                ("05", "Sleep", "recovery · hormones",
                 "7–9 hours. Poor sleep raises ghrelin (hunger hormone) and cuts willpower. If your sleep is off, fix that before optimising anything else — it affects everything downstream."),
                ("06", "Track Progress", "measurement · consistency",
                 "Weigh yourself at the same time each morning. Average it weekly, not daily — weight fluctuates 1–3 lbs from water alone. Take a waist measurement every two weeks as a secondary check."),
            ]

        elif prediction >= LEAN_THRESHOLD:
            status_html = """
            <div style="display:flex;align-items:center;gap:12px;
                        background:rgba(59,130,246,0.14);
                        border:2px solid rgba(59,130,246,0.7);
                        border-left:5px solid #3b82f6;
                        box-shadow:0 4px 20px rgba(59,130,246,0.25);
                        border-radius:12px;padding:14px 18px;margin-bottom:24px;">
                <span style="font-size:22px;color:#93c5fd;">✓</span>
                <div>
                    <div style="font-weight:700;color:#93c5fd;font-size:15px;">Healthy Body Fat Range</div>
                    <div style="color:#94a3b8;font-size:13px;margin-top:2px;">
                        You're in a good place. These recommendations focus on staying here and improving performance.
                    </div>
                </div>
            </div>
            """
            recs = [
                ("01", "Balanced Nutrition", "diet · macros",
                 "No need to be in a deficit. Eat at maintenance and focus on food quality — lean protein, complex carbs, and healthy fats at most meals. Consistency over perfection."),
                ("02", "Strength Training", "muscle · performance",
                 "3–4 lifting sessions per week using progressive overload. Add a little weight or reps each week. Building more muscle shifts your body composition without needing to lose scale weight."),
                ("03", "Cardiovascular Fitness", "endurance · heart health",
                 "2–3 cardio sessions per week. Doesn't need to be intense — a 30-min walk or bike ride counts. Keeps your heart healthy and improves recovery between strength sessions."),
                ("04", "Protein Intake", "diet · muscle retention",
                 f"Keep protein at {round(weight * LBS_TO_KG * 1.6)}–{round(weight * LBS_TO_KG * 2.0)} g per day. You don't need to be obsessive about it — just make sure there's a good protein source at each meal."),
                ("05", "Hydration", "recovery · performance",
                 "About 35 ml per kg of bodyweight daily — more on training days. Thirst is a late signal; don't wait for it. Urine colour is your best real-time indicator (pale yellow is ideal)."),
                ("06", "Body Composition Check-ins", "tracking · awareness",
                 "Weigh in every few weeks and take measurements monthly. You're not trying to change drastically — just catch any gradual drift before it becomes harder to reverse."),
            ]

        else:
            status_html = """
            <div style="display:flex;align-items:center;gap:12px;
                        background:rgba(34,197,94,0.14);
                        border:2px solid rgba(34,197,94,0.7);
                        border-left:5px solid #22c55e;
                        box-shadow:0 4px 20px rgba(34,197,94,0.25);
                        border-radius:12px;padding:14px 18px;margin-bottom:24px;">
                <span style="font-size:22px;color:#86efac;">◆</span>
                <div>
                    <div style="font-weight:700;color:#86efac;font-size:15px;">Lean Body Composition</div>
                    <div style="color:#94a3b8;font-size:13px;margin-top:2px;">
                        Lean is good — but it comes with higher maintenance demands. Recovery becomes the priority.
                    </div>
                </div>
            </div>
            """
            recs = [
                ("01", "Eat Enough", "calories · energy",
                 "At low body fat, chronic undereating suppresses testosterone, disrupts sleep, and stalls muscle growth. Make sure you're eating at or above TDEE on training days — don't chase a lower number."),
                ("02", "High Protein", "diet · muscle",
                 f"Target {round(weight * LBS_TO_KG * 2.0)}–{round(weight * LBS_TO_KG * 2.4)} g of protein daily. At this level of leanness, muscle tissue is more vulnerable to breakdown — protein keeps it protected."),
                ("03", "Prioritise Recovery", "sleep · rest days",
                 "Muscle isn't built during training — it's built during rest. Take 1–2 full rest days per week. 8+ hours of sleep if you're training hard. Overtraining at low body fat leads to injury fast."),
                ("04", "Manage Stress", "hormones · cortisol",
                 "Chronically elevated cortisol eats muscle and deposits fat around the midsection — even in lean individuals. Meditation, breathing work, or just cutting training volume when life is heavy all help."),
                ("05", "Hydration & Electrolytes", "performance · recovery",
                 "At low body fat you carry less water buffer. Stay well hydrated and consider sodium, potassium, and magnesium if you're sweating heavily in training — cramps and fatigue are often electrolyte issues, not fitness ones."),
                ("06", "Periodic Blood Work", "health · monitoring",
                 "Get bloodwork every 6–12 months. Testosterone, thyroid (T3/T4), ferritin, and vitamin D are the ones most likely to drop at sustained low body fat. Catching them early is much easier than fixing them after the fact."),
            ]

        st.markdown(status_html, unsafe_allow_html=True)

        card_html_parts = []
        for num, title, tags, body in recs:
            tag_pills = "".join(
                f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
                f'background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.25);'
                f'color:#94a3b8;font-size:11px;margin-right:6px;margin-bottom:4px;letter-spacing:0.3px;">'
                f'{t.strip()}</span>'
                for t in tags.split("·")
            )
            card_html_parts.append(f"""
<div class="rc">
    <div class="rc-left"><span class="rc-num">{num}</span></div>
    <div class="rc-body">
        <div class="rc-title">{title}</div>
        <div class="rc-tags">{tag_pills}</div>
        <div class="rc-text">{body}</div>
    </div>
</div>
""")

        components.html(f"""
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: transparent; color: #e2e8f0; padding: 4px 0;
}}
::-webkit-scrollbar {{ width:8px; }}
::-webkit-scrollbar-thumb {{ background:#4f46e5; border-radius:8px; }}
.rc {{
    display: flex; gap: 18px; align-items: flex-start;
    background: rgba(17,24,39,0.7);
    border: 2px solid rgba(255,255,255,0.22);
    border-radius: 14px;
    padding: 20px 20px 20px 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 16px rgba(0,0,0,0.35);
}}
.rc:hover {{
    background: rgba(17,24,39,0.9);
    border-color: rgba(96,165,250,0.6);
    box-shadow: 0 6px 24px rgba(96,165,250,0.2);
}}
.rc-left {{ flex-shrink:0; padding-top:2px; }}
.rc-num {{
    display: block; width:32px; height:32px; border-radius:8px;
    background: rgba(255,255,255,0.12);
    border: 2px solid rgba(255,255,255,0.3);
    color: #94a3b8; font-size:11px; font-weight:700;
    letter-spacing:0.5px; text-align:center; line-height:32px;
}}
.rc-body {{ flex:1; min-width:0; }}
.rc-title {{ font-size:15px; font-weight:600; color:#f1f5f9; margin-bottom:6px; letter-spacing:0.1px; }}
.rc-tags  {{ margin-bottom:10px; line-height:1.6; }}
.rc-text  {{ font-size:13.5px; color:#94a3b8; line-height:1.65; }}
</style>
{"".join(card_html_parts)}
""", height=min(len(recs) * 148 + 20, 550), scrolling=True)

    # ── TAB 5 ──────────────────────────────────────────────────────────
    with tab5:
        st.subheader("Calorie & Macro Targets")

        ca, cb = st.columns(2)
        with ca:
            st.metric("BMR",  f"{bmr:.0f} kcal/day",  help="Calories burned at complete rest")
            st.metric("TDEE", f"{tdee:.0f} kcal/day", help="Estimated daily calories burned including activity")
        with cb:
            st.metric("Fat Loss Target",    f"{round(tdee - 400)} kcal/day", "~400 kcal deficit")
            st.metric("Muscle Gain Target", f"{round(tdee + 250)} kcal/day", "~250 kcal surplus")

        st.markdown("---")
        st.subheader("Macro Breakdown")

        weight_kg  = weight * LBS_TO_KG
        protein_g  = round(weight_kg * 2.0)
        fat_g      = round(tdee * 0.28 / 9)
        carb_g     = round((tdee - protein_g * 4 - fat_g * 9) / 4)
        total_kcal = round(tdee)

        macro_fig = go.Figure(go.Pie(
            labels=["Protein", "Fat", "Carbohydrates"],
            values=[protein_g * 4, fat_g * 9, max(carb_g * 4, 0)],
            customdata=[f"{protein_g}g", f"{fat_g}g", f"{int(max(carb_g, 0))}g"],
            hovertemplate="<b>%{label}</b><br>%{value} kcal<br>%{customdata}<extra></extra>",
            hole=0.65,
            marker_colors=["#60a5fa", "#f97316", "#22c55e"],
            marker_line=dict(color="#0b0f1a", width=4),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(color="white", size=14),
            pull=[0.02, 0.02, 0.02],
        ))
        macro_fig.add_annotation(
            text=f"<span style='font-size:26px;font-weight:bold;color:white;'>{total_kcal}</span><br><span style='color:#94a3b8;font-size:14px;'>kcal/day</span>",
            x=0.5, y=0.5, showarrow=False
        )
        macro_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5,
                        font=dict(color="#cbd5e1", size=13)),
            margin=dict(t=30, b=30, l=30, r=30)
        )
        st.plotly_chart(macro_fig, use_container_width=True)

        protein_kcal = protein_g * 4
        fat_kcal     = fat_g * 9
        carb_kcal    = max(carb_g * 4, 0)
        meals        = 4

        def pct(macro_kcal):
            return round(macro_kcal / total_kcal * 100)

        macros_html = ""
        macro_data = [
            ("ti-meat",    "#185FA5", "#E6F1FB", "Protein",      "Muscle repair, satiety, thermogenesis",                  protein_g, protein_kcal),
            ("ti-droplet", "#854F0B", "#FAEEDA", "Fat",           "Hormone synthesis, fat-soluble vitamins, energy reserve", fat_g,     fat_kcal),
            ("ti-grain",   "#3B6D11", "#EAF3DE", "Carbohydrates", "Primary fuel, glycogen storage, brain function",         carb_g,    carb_kcal),
        ]

        for icon, color, bg, label, role, grams, kcal_val in macro_data:
            bar_pct    = pct(kcal_val)
            per_meal_g = round(grams / meals)
            per_meal_k = round(kcal_val / meals)
            macros_html += f"""
            <div class="macro-card">
            <div class="macro-row">
                <div class="macro-icon" style="background:{bg}">
                <i class="ti {icon}" style="color:{color};font-size:17px" aria-hidden="true"></i>
                </div>
                <div>
                <div class="macro-label">{label}
                    <span class="split-badge">{bar_pct}% of calories</span>
                </div>
                <div class="macro-role">{role}</div>
                </div>
                <div class="macro-right">
                <div class="macro-grams">{grams} g</div>
                <div class="macro-kcal">{kcal_val} kcal</div>
                </div>
            </div>
            <div class="bar-row">
                <div></div>
                <div class="bar-bg">
                <div class="bar-fill" style="width:{bar_pct}%;background:{color}"></div>
                </div>
                <div class="bar-pct">{bar_pct}%</div>
            </div>
            <div class="divider"></div>
            <div class="meal-row">
                <div></div>
                <div>
                <span class="meal-label">Per meal ({meals}x/day) &nbsp;</span>
                <span class="meal-val">{per_meal_g} g &nbsp;&middot;&nbsp; {per_meal_k} kcal</span>
                </div>
            </div>
            </div>
            """

        components.html(f"""
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
        <style>
        body{{margin:0;font-family:system-ui,sans-serif;color:#e2e8f0;background:transparent}}
        .macro-card{{
            background:rgba(17,24,39,.85);
            border:2px solid rgba(255,255,255,.28);
            border-radius:14px;overflow:hidden;margin-bottom:10px;
            box-shadow:0 4px 18px rgba(0,0,0,.4);
        }}
        .macro-row{{display:grid;grid-template-columns:2.2rem 1fr auto;
                    align-items:center;gap:0 14px;padding:14px 16px 8px}}
        .macro-icon{{width:2.2rem;height:2.2rem;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0}}
        .macro-label{{font-size:15px;font-weight:500}}
        .macro-role{{font-size:12px;color:#94a3b8;margin-top:2px}}
        .macro-right{{text-align:right}}
        .macro-grams{{font-size:18px;font-weight:500}}
        .macro-kcal{{font-size:12px;color:#94a3b8;margin-top:1px}}
        .bar-row{{padding:0 16px 12px;display:grid;
                    grid-template-columns:2.2rem 1fr auto;gap:0 14px;align-items:center}}
        .bar-bg{{background:rgba(255,255,255,.18);border-radius:999px;height:7px;overflow:hidden}}
        .bar-fill{{height:100%;border-radius:999px}}
        .bar-pct{{font-size:12px;color:#94a3b8;text-align:right;min-width:32px}}
        .meal-row{{padding:0 16px 12px;display:grid;grid-template-columns:2.2rem 1fr;gap:0 14px}}
        .meal-label{{font-size:11px;color:#64748b}}
        .meal-val{{font-size:12px;color:#94a3b8}}
        .divider{{height:1px;background:rgba(255,255,255,.2);margin:0 16px}}
        .total-card{{
            background:rgba(30,41,59,.8);
            border:2px solid rgba(255,255,255,.28);
            border-radius:14px;padding:14px 16px;display:flex;
            align-items:center;justify-content:space-between;margin-top:4px;
            box-shadow:0 4px 18px rgba(0,0,0,.4);
        }}
        .total-label{{font-size:14px;color:#94a3b8}}
        .total-val{{font-size:20px;font-weight:500}}
        .meal-note{{font-size:11px;color:#64748b;margin-top:2px}}
        .split-badge{{font-size:11px;background:rgba(99,102,241,.25);color:#c4b5fd;
                      padding:3px 8px;border-radius:999px;margin-left:8px;vertical-align:middle;
                      border:1px solid rgba(99,102,241,.5);}}
        </style>
        <div style="padding:0.5rem 0">
        {macros_html}
        <div class="total-card">
            <div>
            <div class="total-label">Daily total</div>
            <div class="meal-note">≈ {round(total_kcal / meals):,} kcal per meal over {meals} meals</div>
            </div>
            <div class="total-val">{total_kcal:,} kcal / day</div>
        </div>
        </div>
        """, height=len(macro_data) * 140 + 90, scrolling=False)

        st.caption("*Harris-Benedict BMR + activity multiplier. Consult a nutritionist for personalised plans.*")

    # ── TAB 6 ──────────────────────────────────────────────────────────
    with tab6:
        st.subheader("Prediction History — This Session")

        if not st.session_state.history:
            st.info("No predictions yet. Run at least one prediction to see history.")
        else:
            hist_df      = pd.DataFrame(st.session_state.history)
            display_cols = ["Timestamp", "Body Fat %", "BMI", "WHR",
                            "BMR (kcal)", "TDEE (kcal)", "Health Score", "Category"]

            if len(hist_df) > 1:
                trend_fig = px.line(
                    hist_df, x="Timestamp", y="Body Fat %",
                    markers=True, title="Body Fat % Over Predictions",
                )
                trend_fig.update_traces(line_color="#60a5fa", line_width=3, marker_color="#a78bfa", marker_size=9)
                trend_fig.update_layout(paper_bgcolor="#0b0f1a", plot_bgcolor="#0b0f1a",
                                        font_color="white", height=280)
                st.plotly_chart(trend_fig, use_container_width=True)

                @st.fragment
                def render_comparison(df):
                    st.subheader("Multi-Metric Comparison")
                    compare_metric = st.selectbox(
                        "Compare across predictions:",
                        ["Body Fat %", "BMI", "Health Score", "WHR"],
                    )
                    cscale_map = {
                        "Body Fat %": "Blues", "BMI": "Greens",
                        "Health Score": "Purples", "WHR": "Oranges"
                    }
                    bar_fig = px.bar(
                        df, x="Timestamp", y=compare_metric,
                        color=compare_metric, color_continuous_scale=cscale_map[compare_metric],
                        title=f"{compare_metric} Across Predictions",
                    )
                    bar_fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white", height=300, margin=dict(t=50, b=20, l=20, r=20)
                    )
                    st.plotly_chart(bar_fig, use_container_width=True)

                render_comparison(hist_df)

            st.dataframe(
                hist_df[display_cols].style.background_gradient(
                    subset=["Body Fat %"], cmap="RdYlGn_r"
                ),
                use_container_width=True,
            )

            all_csv = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download History CSV", all_csv, "body_fat_history.csv", "text/csv")

            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    st.markdown("""
    <div style="margin-top:30px;padding:14px 10px;text-align:center;
                border-top:3px solid rgba(96,165,250,.4);
                color:#94a3b8;font-size:12px;">
        ◆ <b>Body Fat System</b> • Powered by Machine Learning<br>
        <span style="color:#64748b;font-size:11px;">
            Built for analysis, tracking, and educational insights only
        </span>
    </div>
    """, unsafe_allow_html=True)