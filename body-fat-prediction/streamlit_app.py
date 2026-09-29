import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from particle_background import render_particle_background

st.set_page_config(
    page_title="Body Fat Estimate",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────
LBS_TO_KG: float = 0.453592
IN_TO_M:   float = 0.0254
IN_TO_CM:  float = 2.54

LEAN_THRESHOLD:    float = 14.0
HEALTHY_THRESHOLD: float = 25.0

ACTIVITY_LEVELS: dict[str, float] = {
    "Sedentary (desk job)":              1.200,
    "Lightly active (1-3 days/week)":    1.375,
    "Moderately active (3-5 days/week)": 1.550,
    "Very active (6-7 days/week)":       1.725,
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

# ── Design tokens (one accent, zinc neutrals) ────────────────────────
INK    = "#ececee"
MUTED  = "#8d9097"
FAINT  = "#6b6e75"
LINE   = "#2a2c31"
ACCENT = "#e9b44c"

LEAN_C    = "#6cc59a"
HEALTHY_C = ACCENT
HIGH_C    = "#ef6f55"

FONT_SANS = "'Hanken Grotesk', system-ui, -apple-system, 'Segoe UI', sans-serif"
FONT_MONO = FONT_SANS

if "history" not in st.session_state:
    st.session_state.history = []


# ── Rendering helpers ────────────────────────────────────────────────
def html(markup: str) -> None:
    """Render raw HTML. Strips indentation and blank lines so Markdown
    never mistakes indented HTML for a code block."""
    lines = (line.strip() for line in markup.strip().splitlines())
    st.markdown("\n".join(l for l in lines if l), unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, color=INK, size=13),
        height=height,
        margin=layout.pop("margin", dict(t=16, b=16, l=8, r=8)),
        hoverlabel=dict(bgcolor="#15171a", bordercolor=LINE,
                        font=dict(family=FONT_SANS, color=INK)),
        **layout,
    )
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, width="stretch",
                    config={"displayModeBar": False})


def inject_css() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700&display=swap');

.block-container {
    max-width: 1320px;
    padding-top: 2.75rem;
    padding-bottom: 4rem;
}
html, body { background: #0e0f11; }
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"] { background: transparent !important; }
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p { color: #111316 !important; font-weight: 600; }
::selection { background: #e9b44c; color: #111316; }

.num {
    font-variant-numeric: tabular-nums lining-nums;
    letter-spacing: -0.01em;
}

/* Profile panel (right) */
@media (min-width: 761px) {
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .profile-anchor) { flex-direction: row-reverse; align-items: flex-start; }
  [data-testid="stColumn"]:has(.profile-anchor) { position: sticky; top: 1rem; max-height: calc(100vh - 2rem); overflow-y: auto; }
}
[data-testid="stColumn"]:has(.profile-anchor) { background: #131518; border: 1px solid #2a2c31; border-radius: 8px; padding: 1.1rem 1.1rem 1.25rem; }
[data-testid="stColumn"]:has(.profile-anchor) h3 { font-size: 1rem; font-weight: 600; margin: .25rem 0 .5rem; padding: 0; }

/* Page header */
.page-head { margin-bottom: 2.25rem; max-width: 62ch; }
.page-head h1 {
    font-size: 2rem; font-weight: 600; letter-spacing: -0.025em;
    line-height: 1.15; margin: 0 0 .5rem; padding: 0; color: #ececee;
}
.page-head p { color: #a9abb1; font-size: 1rem; line-height: 1.6; margin: 0; }

/* Empty state */
.empty {
    border: 1px dashed #34373c; border-radius: 8px;
    padding: 2.5rem 2rem; background: #15171a;
}
.empty h3 { font-size: 1.05rem; font-weight: 600; margin: 0 0 .5rem; padding: 0; }
.empty p { color: #a9abb1; margin: 0 0 .25rem; line-height: 1.6; max-width: 60ch; }
.facts {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 1.5rem; margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #2a2c31;
}
.facts .k { font-size: .8rem; color: #8d9097; margin-bottom: .25rem; }
.facts .v { font-size: 1.35rem; font-weight: 500; color: #ececee; }

/* Result summary */
.result {
    display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
    gap: 2.5rem; align-items: end;
    padding-bottom: 1.75rem; border-bottom: 1px solid #2a2c31; margin-bottom: 1.75rem;
}
.result-label { font-size: .85rem; color: #8d9097; margin-bottom: .35rem; }
.result-value { font-size: 4.25rem; font-weight: 500; line-height: 1; letter-spacing: -0.04em; }
.result-value span { font-size: 2rem; margin-left: .15rem; color: #8d9097; }
.result-cat { margin-top: .75rem; color: #a9abb1; font-size: .95rem; }
.result-cat b { font-weight: 600; }
.goal {
    background: #15171a; border: 1px solid #2a2c31; border-radius: 8px;
    padding: 1rem 1.15rem; color: #c9cace; font-size: .92rem; line-height: 1.55;
}
.goal .k { font-size: .8rem; color: #8d9097; margin-bottom: .3rem; }

/* Stat grid */
.stats {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    column-gap: 2rem; row-gap: 1.5rem; margin-bottom: 2.25rem;
}
.stat { border-top: 1px solid #2a2c31; padding-top: .8rem; }
.stat-label { font-size: .8rem; color: #8d9097; }
.stat-value { font-size: 1.6rem; font-weight: 500; color: #ececee; margin-top: .2rem; }
.stat-note { font-size: .82rem; color: #8d9097; margin-top: .15rem; }

/* Section text */
.section-note { color: #a9abb1; font-size: .92rem; line-height: 1.6; max-width: 65ch; margin: -.25rem 0 1rem; }
.legend { display: flex; flex-wrap: wrap; gap: 1.25rem; font-size: .8rem; color: #a9abb1; margin: -.25rem 0 1.25rem; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: .4rem; vertical-align: -1px; }

/* Flag rows */
.flags { display: grid; gap: 0; margin-bottom: 1.5rem; }
.flag { display: grid; grid-template-columns: 4px 1fr; gap: 1rem; padding: .9rem 0; border-bottom: 1px solid #2a2c31; }
.flag:first-child { border-top: 1px solid #2a2c31; }
.flag .bar { border-radius: 2px; }
.flag .t { font-weight: 600; font-size: .95rem; color: #ececee; }
.flag .d { color: #a9abb1; font-size: .9rem; line-height: 1.55; margin-top: .15rem; }

/* Measurement grid */
.mgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 1px; background: #2a2c31; border: 1px solid #2a2c31; border-radius: 8px; overflow: hidden; margin-bottom: 2rem; }
.mcell { background: #15171a; padding: 1rem 1.1rem 1.1rem; }
.mcell .name { font-size: .82rem; color: #8d9097; }
.mcell .val { font-size: 1.35rem; font-weight: 500; margin: .15rem 0 .1rem; }
.mcell .val small { font-size: .8rem; color: #8d9097; margin-left: .15rem; }
.mcell .status { font-size: .8rem; font-weight: 500; }
.track { position: relative; height: 14px; margin-top: .7rem; }
.track .axis { position: absolute; left: 0; right: 0; top: 6px; height: 2px; background: #22252a; }
.track .band { position: absolute; top: 4px; height: 6px; background: #4a3d22; border-radius: 3px; }
.track .dot  { position: absolute; top: 2px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%; border: 2px solid #15171a; box-shadow: 0 0 0 1px rgba(0,0,0,.45); }
.mcell .range { font-size: .75rem; color: #6b6e75; margin-top: .25rem; }

/* Recommendations */
.rec-status { border-top: 1px solid; padding-top: .9rem; margin-bottom: 1.75rem; }
.rec-status .t { font-weight: 600; font-size: 1rem; }
.rec-status .d { color: #a9abb1; font-size: .92rem; margin-top: .2rem; line-height: 1.55; }
.recs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: 2.5rem; row-gap: 1.75rem; }
.rec h4 { font-size: .98rem; font-weight: 600; margin: 0 0 .35rem; padding: 0; color: #ececee; }
.rec p { color: #a9abb1; font-size: .9rem; line-height: 1.6; margin: 0; }

/* Macro table */
.macros { border: 1px solid #2a2c31; border-radius: 8px; overflow: hidden; background: #15171a; }
.macro { display: grid; grid-template-columns: 12px 1fr auto auto; gap: 1.25rem; align-items: center; padding: .9rem 1.1rem; border-bottom: 1px solid #2a2c31; }
.macro .sw { width: 12px; height: 12px; border-radius: 3px; }
.macro .n { font-weight: 600; font-size: .95rem; }
.macro .r { color: #8d9097; font-size: .82rem; margin-top: .1rem; }
.macro .g { font-size: 1.1rem; font-weight: 500; text-align: right; }
.macro .s { color: #8d9097; font-size: .8rem; text-align: right; min-width: 7.5rem; }
.macro-total { display: flex; justify-content: space-between; align-items: baseline; padding: .9rem 1.1rem; background: #111316; }
.macro-total .k { color: #a9abb1; font-size: .9rem; }
.macro-total .v { font-size: 1.15rem; font-weight: 500; }

/* Model note */
.model-note { border-top: 1px solid #2a2c31; padding-top: 1.25rem; margin-top: 1.5rem; color: #a9abb1; font-size: .9rem; line-height: 1.65; max-width: 70ch; }
.model-note b { color: #ececee; font-weight: 600; }

/* Sidebar guide */
.guide dt { font-weight: 600; font-size: .85rem; color: #ececee; margin-top: .7rem; }
.guide dd { margin: .1rem 0 0; font-size: .82rem; color: #a9abb1; line-height: 1.5; }
.guide .grp { font-size: .75rem; color: #8d9097; margin-top: 1.1rem; padding-top: .6rem; border-top: 1px solid #2a2c31; }
.guide .grp:first-child { margin-top: 0; padding-top: 0; border-top: 0; }
.guide .tip { margin-top: 1rem; font-size: .82rem; color: #c9cace; line-height: 1.5; }

.foot { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #2a2c31; color: #8d9097; font-size: .8rem; }

/* Tabs: quieter */
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
.stTabs [data-baseweb="tab"] { padding-left: 0; padding-right: 0; }

@media (max-width: 760px) {
    .result, .recs { grid-template-columns: 1fr; gap: 1.25rem; }
    .stats, .facts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .result-value { font-size: 3.25rem; }
    .macro { grid-template-columns: 12px 1fr auto; }
    .macro .s { display: none; }
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────
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
            "Run `streamlit run streamlit_app.py` from the folder containing `models/` and `notebook/`."
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


# ── Domain calculations (unchanged) ──────────────────────────────────
def classify_body_fat(pct: float) -> tuple[str, str]:
    if pct < LEAN_THRESHOLD:
        return "LEAN",          LEAN_C
    if pct < HEALTHY_THRESHOLD:
        return "HEALTHY",       HEALTHY_C
    return "HIGH BODY FAT",     HIGH_C


@st.cache_data
def compute_bmi(weight_lbs: float, height_in: float) -> float:
    h_m = height_in * IN_TO_M
    return (weight_lbs * LBS_TO_KG) / (h_m ** 2)


def classify_bmi(bmi: float) -> tuple[str, str]:
    if bmi < 18.5: return "Underweight", "#d9a441"
    if bmi < 25.0: return "Normal",      LEAN_C
    if bmi < 30.0: return "Overweight",  "#f0955a"
    return "Obese", HIGH_C


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
    return bmr * ACTIVITY_LEVELS.get(level, 1.375)


def body_fat_percentile(pct: float) -> int:
    breakpoints = [
        (5, 1), (10, 5), (14, 15), (18, 35), (22, 55),
        (25, 65), (30, 80), (35, 90), (40, 96), (50, 99),
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
    return round(abs(delta_lbs) / weekly_rate, 1)


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


# ── Chart builders ───────────────────────────────────────────────────
def scale_chart(value: float, rng: tuple[float, float], steps: list[tuple],
                color: str, suffix: str = "") -> go.Figure:
    """Horizontal bullet scale: tinted bands, a thin value bar, a marker."""
    fig = go.Figure(go.Indicator(
        mode="number+gauge",
        value=round(value, 1),
        number={"suffix": suffix, "font": {"family": FONT_MONO, "size": 30, "color": INK}},
        domain={"x": [0.2, 1], "y": [0, 1]},
        gauge={
            "shape": "bullet",
            "axis": {"range": list(rng), "tickcolor": LINE,
                     "tickfont": {"family": FONT_MONO, "size": 11, "color": MUTED}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#15171a",
            "borderwidth": 0,
            "steps": [{"range": [a, b], "color": c} for a, b, c in steps],
            "threshold": {"line": {"color": INK, "width": 2},
                          "thickness": 0.9, "value": value},
        },
    ))
    return style_fig(fig, 110, margin=dict(t=10, b=30, l=10, r=20))


inject_css()
render_particle_background(
    dot="rgba(236,236,238,0.35)", link="#ececee", near="#e9b44c",
)


# -- Layout: results on the left, profile panel on the right --
profile_col, main_col = st.columns([1.25, 2.4], gap="large")
profile_col.markdown('<span class="profile-anchor"></span>', unsafe_allow_html=True)

# -- Profile panel: inputs --
profile_col.markdown("### Your profile")

sex = profile_col.selectbox("Sex", ["Male", "Female"])
age = profile_col.number_input("Age", 1, 100, 25)
profile_col.caption("Used for metabolic rate and reference ranges.")

activity = profile_col.selectbox("Daily activity", list(ACTIVITY_LEVELS.keys()), index=1)
goal_bf = profile_col.slider("Target body fat %", 5, 40, 18,
                            help="Used to estimate how many weeks it takes to reach this goal.")

col1, col2 = profile_col.columns(2)
with col1:
    weight = st.number_input("Weight (lbs)", 50, 400, 170)
with col2:
    height = st.number_input("Height (in)", 40, 90, 68)

bmi_preview = compute_bmi(weight, height)
profile_col.caption(f"BMI from these values: **{bmi_preview:.1f}**")

profile_col.divider()
profile_col.markdown("### Measurements (cm)")

with profile_col.expander("How to measure", expanded=False):
    html("""
    <dl class="guide">
      <div class="grp">Upper body</div>
      <dt>Neck</dt><dd>Just below the larynx, tape relaxed.</dd>
      <dt>Chest</dt><dd>At nipple level, arms relaxed, breathing normally.</dd>
      <dt>Biceps</dt><dd>Midpoint of the upper arm. Flexed or relaxed, but the same every time.</dd>
      <dt>Forearm</dt><dd>Widest point, hand open and relaxed.</dd>
      <div class="grp">Core</div>
      <dt>Abdomen</dt><dd>At navel level, stomach relaxed. Do not suck in.</dd>
      <dt>Hip</dt><dd>Widest part of the buttocks, feet together.</dd>
      <div class="grp">Lower body</div>
      <dt>Thigh</dt><dd>Upper thigh, just below the glutes.</dd>
      <dt>Knee</dt><dd>Around the kneecap, standing naturally.</dd>
      <dt>Ankle</dt><dd>Narrowest point above the ankle bone.</dd>
      <dt>Wrist</dt><dd>Just below the wrist bone, hand relaxed.</dd>
      <div class="tip">Measure at the same time of day, ideally in the morning and not after a meal or workout. Consistency matters more than precision.</div>
    </dl>
    """)

with profile_col.expander("Upper body", expanded=True):
    neck    = st.slider("Neck",    20,  60,  38)
    chest   = st.slider("Chest",   50, 180, 100)
    biceps  = st.slider("Biceps",  15,  60,  32)
    forearm = st.slider("Forearm", 15,  50,  28)
    wrist   = st.slider("Wrist",   10,  30,  18)

with profile_col.expander("Core", expanded=True):
    abdomen = st.slider("Abdomen", 50, 180, 90)
    hip     = st.slider("Hip",     50, 180, 95)

with profile_col.expander("Lower body", expanded=True):
    thigh = st.slider("Thigh", 20, 100, 55)
    knee  = st.slider("Knee",  20,  70, 40)
    ankle = st.slider("Ankle", 10,  40, 22)

predict_clicked = profile_col.button(
    "Estimate body fat",
    type="primary",
    disabled=(model is None),
    width="stretch",
)


with main_col:
    # ── Page header ──────────────────────────────────────────────────────
    html("""
    <div class="page-head">
      <h1>Body fat estimate</h1>
      <p>Estimate your body fat percentage from a tape measure, a scale and your age,
      using a linear regression model trained on underwater weighing data.</p>
    </div>
    """)

    if load_error:
        st.error(
            f"Could not load model files: {load_error}\n\n"
            "Make sure the `models/` and `notebook/` folders sit next to `streamlit_app.py`."
        )


    # ── Empty state ──────────────────────────────────────────────────────
    if not predict_clicked:
        runs = len(st.session_state.history)
        runs_line = (
            f"<p>You have {runs} estimate{'s' if runs != 1 else ''} this session. "
            "Run another to see how they compare.</p>" if runs else ""
        )
        n_features = len(features) if features is not None else 13
        html(f"""
        <div class="empty">
          <h3>Enter your measurements to get an estimate</h3>
          <p>Fill in your profile and ten body measurements in the profile panel, then press
          <b>Estimate body fat</b>. The measuring guide in that panel shows where to place the tape.</p>
          {runs_line}
          <div class="facts">
            <div><div class="k">Model</div><div class="v">Linear regression</div></div>
            <div><div class="k">Inputs</div><div class="v num">{n_features}</div></div>
            <div><div class="k">Output range</div><div class="v num">0-60%</div></div>
          </div>
        </div>
        """)


    # ── Results ──────────────────────────────────────────────────────────
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
            st.error(f"Feature mismatch: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Preprocessing error: {exc}")
            st.stop()

        _infer_start = time.perf_counter()
        prediction   = float(np.clip(model.predict(input_scaled)[0], 0, 60))
        _infer_ms    = round((time.perf_counter() - _infer_start) * 1000, 2)

        category, category_color = classify_body_fat(prediction)
        category_title = category.capitalize() if category != "HIGH BODY FAT" else "High"

        bmi              = compute_bmi(weight, height)
        bmi_label, bmi_color = classify_bmi(bmi)
        whr              = compute_whr(abdomen, hip)
        bmr              = compute_bmr(weight, height, age, sex)
        tdee             = tdee_from_activity(bmr, activity)
        health_score     = max(0.0, 100 - prediction * 2)
        percentile       = body_fat_percentile(prediction)
        fat_mass_lbs     = round(weight * prediction / 100, 1)
        lean_mass_lbs    = round(weight - fat_mass_lbs, 1)
        weeks            = weeks_to_goal(prediction, goal_bf, weight, tdee)

        record = build_record(prediction, bmi, whr, bmr, tdee,
                              health_score, category, input_dict)
        st.session_state.history.append(record)

        is_male  = (sex == "Male")
        whr_high = 0.90 if is_male else 0.85
        whr_mod  = 0.85 if is_male else 0.80
        abd_high = 102  if is_male else 88

        # Goal sentence
        if weeks == 0.0:
            goal_text = f"You are already at your {goal_bf}% target."
        elif prediction > goal_bf:
            goal_text = (f"Reaching <b>{goal_bf}%</b> takes about <b class='num'>{weeks}</b> weeks "
                         "on a 500 kcal/day deficit at your current weight.")
        else:
            goal_text = (f"Reaching <b>{goal_bf}%</b> takes about <b class='num'>{weeks}</b> weeks "
                         "on a 250 kcal/day surplus at your current weight.")

        diff_vs_ref = prediction - 20
        band_text = {
            "LEAN":          "below the 14% line",
            "HEALTHY":       "inside the 14-25% band",
            "HIGH BODY FAT": "above the 25% line",
        }[category]

        html(f"""
        <section class="result">
          <div>
            <div class="result-label">Estimated body fat</div>
            <div class="result-value num" style="color:{category_color}">{prediction:.1f}<span>%</span></div>
            <div class="result-cat"><b style="color:{category_color}">{category_title}</b>,
              {band_text}. <span class="num">{diff_vs_ref:+.1f}</span> points against a 20% reference.</div>
          </div>
          <div class="goal">
            <div class="k">Target: {goal_bf}%</div>
            {goal_text}
          </div>
        </section>
        """)

        whr_note = "Above risk line" if whr > whr_high else "Normal"
        html(f"""
        <div class="stats">
          <div class="stat"><div class="stat-label">BMI</div>
            <div class="stat-value num">{bmi:.1f}</div><div class="stat-note">{bmi_label}</div></div>
          <div class="stat"><div class="stat-label">Waist-to-hip ratio</div>
            <div class="stat-value num">{whr:.2f}</div><div class="stat-note">{whr_note}</div></div>
          <div class="stat"><div class="stat-label">Health score</div>
            <div class="stat-value num">{health_score:.0f}<span style="color:{FAINT};font-size:1rem">/100</span></div>
            <div class="stat-note">100 minus twice your body fat %</div></div>
          <div class="stat"><div class="stat-label">Fat mass</div>
            <div class="stat-value num">{fat_mass_lbs}</div><div class="stat-note">lbs</div></div>
          <div class="stat"><div class="stat-label">Lean mass</div>
            <div class="stat-value num">{lean_mass_lbs}</div><div class="stat-note">lbs</div></div>
          <div class="stat"><div class="stat-label">Daily energy (TDEE)</div>
            <div class="stat-value num">{tdee:,.0f}</div><div class="stat-note">kcal per day</div></div>
        </div>
        """)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Predictions", "Health", "Impact", "Recommendations", "Nutrition", "History",
        ])

        # ── Predictions ──────────────────────────────────────────────────
        with tab1:
            st.markdown("#### Where you sit")

            show_chart(scale_chart(
                prediction, (0, 60),
                [(0, 14, "#1d3a2c"), (14, 25, "#3d3320"), (25, 60, "#43241d")],
                category_color, "%",
            ))
            html(f"""
            <div class="legend">
              <span><i style="background:#1d3a2c;border:1px solid {LEAN_C}"></i>Lean, under 14%</span>
              <span><i style="background:#3d3320;border:1px solid {HEALTHY_C}"></i>Healthy, 14-25%</span>
              <span><i style="background:#43241d;border:1px solid {HIGH_C}"></i>High, over 25%</span>
            </div>
            """)

            show_chart(scale_chart(
                bmi, (10, 50),
                [(10, 18.5, "#3a331f"), (18.5, 25, "#1d3a2c"),
                 (25, 30, "#3f2a1d"), (30, 50, "#43241d")],
                bmi_color,
            ))
            html("""
            <div class="legend">
              <span><i style="background:#3a331f"></i>Underweight, under 18.5</span>
              <span><i style="background:#1d3a2c"></i>Normal, 18.5-24.9</span>
              <span><i style="background:#3f2a1d"></i>Overweight, 25-29.9</span>
              <span><i style="background:#43241d"></i>Obese, 30 and over</span>
            </div>
            """)

            st.markdown("#### Measurements against a reference")
            html('<p class="section-note">Six circumferences compared with a typical healthy adult. '
                 'Points outside the grey outline are larger than the reference.</p>')

            radar_keys = ["Abdomen", "Chest", "Hip", "Thigh", "Biceps", "Forearm"]
            user_vals  = [input_dict.get(k, 0) for k in radar_keys]
            ref_vals   = [MEASUREMENT_META[k]["ref"] for k in radar_keys]

            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=ref_vals + [ref_vals[0]], theta=radar_keys + [radar_keys[0]],
                name="Reference", line=dict(color=FAINT, width=1.5, dash="dot"),
                fill="toself", fillcolor="rgba(236,236,238,0.05)",
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=user_vals + [user_vals[0]], theta=radar_keys + [radar_keys[0]],
                name="You", line=dict(color=ACCENT, width=2),
                fill="toself", fillcolor="rgba(233,180,76,0.14)",
                marker=dict(size=6, color=ACCENT),
            ))
            style_fig(
                radar_fig, 440,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor=LINE, linecolor=LINE,
                                    tickfont=dict(family=FONT_MONO, size=10, color=FAINT)),
                    angularaxis=dict(gridcolor=LINE, linecolor=LINE,
                                     tickfont=dict(size=12, color=INK)),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                            xanchor="center", x=0.5, font=dict(color=MUTED)),
                margin=dict(t=30, b=50, l=40, r=40),
            )
            show_chart(radar_fig)

            csv_bytes = pd.DataFrame([record]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download this result (CSV)",
                data=csv_bytes,
                file_name="body_fat_result.csv",
                mime="text/csv",
            )

        # ── Health ───────────────────────────────────────────────────────
        with tab2:
            st.markdown("#### Risk markers")

            whr_risk = (
                "high risk"     if whr > whr_high else
                "moderate risk" if whr > whr_mod else
                "low risk"
            )
            html(f'<p class="section-note">Waist-to-hip ratio <b class="num">{whr}</b>: {whr_risk} '
                 f'for {sex.lower()}s. Your body fat puts you around the '
                 f'<b class="num">{percentile}th</b> percentile of the population.</p>')

            flags = []
            if abdomen > abd_high:
                flags.append((HIGH_C, "High abdomen circumference",
                              f"Your abdomen ({abdomen} cm) is above the {abd_high} cm threshold, a strong predictor of visceral fat."))
            if weight > 180:
                flags.append(("#f0955a", "Higher weight profile",
                              "Weight contributes significantly to the overall fat mass estimate."))
            if age > 40:
                flags.append(("#d9a441", "Age factor",
                              "Metabolism slows past 40. Resistance training is the main lever for keeping muscle."))
            if thigh > 60:
                flags.append((ACCENT, "Large thigh measurement",
                              "A larger thigh measurement has a moderate upward influence on the estimate."))
            if whr > whr_high:
                flags.append((HIGH_C, "Elevated waist-to-hip ratio",
                              f"Above {whr_high} indicates elevated cardiovascular risk for {sex.lower()}s."))
            elif whr > whr_mod:
                flags.append(("#f0955a", "Moderate waist-to-hip ratio",
                              f"Above {whr_mod} indicates moderate cardiovascular risk."))
            if bmi > 30:
                flags.append((HIGH_C, "BMI in the obese range",
                              "Consider seeing a healthcare provider for a clinical assessment."))
            if not flags:
                flags.append((LEAN_C, "No risk markers",
                              "None of your measurements cross the thresholds this app checks."))

            rows = "".join(
                f'<div class="flag"><div class="bar" style="background:{c}"></div>'
                f'<div><div class="t">{t}</div><div class="d">{d}</div></div></div>'
                for c, t, d in flags
            )
            html(f'<div class="flags">{rows}</div>')

            st.markdown(f"#### ACE body fat categories for {sex.lower()}s")

            if is_male:
                brackets = [("Essential", 0, 5), ("Athletes", 5, 13), ("Fitness", 13, 17),
                            ("Average", 17, 24), ("Obese", 24, 60)]
            else:
                brackets = [("Essential", 0, 13), ("Athletes", 13, 20), ("Fitness", 20, 24),
                            ("Average", 24, 31), ("Obese", 31, 60)]

            range_fig = go.Figure()
            for i, (label, x0, x1) in enumerate(brackets):
                inside = x0 <= prediction < x1 or (x1 == 60 and prediction >= 60)
                fill = "#4a3d22" if inside else ("#191b1f" if i % 2 == 0 else "#1d2024")
                range_fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1,
                                    fillcolor=fill, line=dict(color="#0e0f11", width=2))
                range_fig.add_annotation(x=(x0 + x1) / 2, y=0.5, text=label, showarrow=False,
                                         font=dict(size=11, color=INK if inside else MUTED))
            range_fig.add_shape(type="line", x0=prediction, x1=prediction, y0=-0.05, y1=1.05,
                                line=dict(color=ACCENT, width=2))
            range_fig.add_annotation(x=prediction, y=1.22, text=f"You, {prediction:.1f}%",
                                     showarrow=False, font=dict(family=FONT_MONO, size=12, color=ACCENT))
            style_fig(
                range_fig, 150, margin=dict(t=30, b=30, l=8, r=8),
                xaxis=dict(range=[0, 60], showgrid=False, zeroline=False, ticksuffix="%",
                           tickfont=dict(family=FONT_MONO, size=11, color=MUTED)),
                yaxis=dict(visible=False, range=[-0.1, 1.35]),
            )
            show_chart(range_fig)

            html(f"""
            <div class="model-note">
              <b>About the model.</b> A linear regression trained on body measurements
              (circumferences, weight, height and age). Inputs are standardised before prediction and the
              output is clipped to 0-60%. This estimate took <span class="num">{_infer_ms} ms</span> to compute.
              <br><br>
              This tool is for education and self-tracking. It is not a clinical assessment, so talk to a
              healthcare professional before acting on it.
            </div>
            """)

        # ── Impact ───────────────────────────────────────────────────────
        with tab3:
            st.markdown("#### Your measurements")
            html('<p class="section-note">The shaded band is the typical healthy range for each site. '
                 'The dot is you.</p>')

            user_values = {
                "Neck": neck, "Chest": chest, "Abdomen": abdomen, "Hip": hip,
                "Thigh": thigh, "Knee": knee, "Ankle": ankle,
                "Biceps": biceps, "Forearm": forearm, "Wrist": wrist,
            }

            cells = ""
            for feature, meta in MEASUREMENT_META.items():
                low, high = meta["ideal"]
                val = user_values[feature]
                if val < low:
                    status, color = "Below range", "#d9a441"
                elif val > high:
                    status, color = "Above range", HIGH_C
                else:
                    status, color = "In range", LEAN_C

                lo_ax = min(low, val) * 0.85
                hi_ax = max(high, val) * 1.10
                span  = hi_ax - lo_ax
                band_l = (low - lo_ax) / span * 100
                band_w = (high - low) / span * 100
                dot_l  = (val - lo_ax) / span * 100

                cells += f"""
                <div class="mcell">
                  <div class="name">{feature}</div>
                  <div class="val num">{val}<small>cm</small></div>
                  <div class="status" style="color:{color}">{status}</div>
                  <div class="track">
                    <div class="axis"></div>
                    <div class="band" style="left:{band_l:.1f}%;width:{band_w:.1f}%"></div>
                    <div class="dot" style="left:{dot_l:.1f}%;background:{color}"></div>
                  </div>
                  <div class="range num">{low}-{high} cm</div>
                </div>
                """
            html(f'<div class="mgrid">{cells}</div>')

            st.markdown("#### What drives the estimate")
            coeff_path = os.path.join(MODEL_DIR, "coefficients.csv")
            if not os.path.exists(coeff_path):
                st.warning("`models/coefficients.csv` not found. Export it from the training notebook "
                           "into `models/` to see feature contributions.")
            else:
                coeff_df = pd.read_csv(coeff_path)
                coeff_df["Percentage"] = (coeff_df["AbsoluteCoefficient"]
                                          / coeff_df["AbsoluteCoefficient"].sum()) * 100
                table_df = (coeff_df.sort_values("Percentage", ascending=False)
                                    .reset_index(drop=True))
                table_df["Rank"]    = range(1, len(table_df) + 1)
                table_df["Feature"] = table_df["Feature"].str.replace("_", " ")
                table_df["Impact"]  = table_df["Percentage"].apply(
                    lambda x: "High" if x > 10 else "Medium" if x > 5 else "Low")

                html('<p class="section-note">Share of the total absolute coefficient weight, on '
                     'standardised inputs. Abdomen dominates, which matches the underlying research.</p>')
                st.dataframe(
                    table_df[["Rank", "Feature", "Percentage", "Impact"]],
                    hide_index=True,
                    width="stretch",
                    height=min(len(table_df) * 35 + 38, 520),
                    column_config={
                        "Rank": st.column_config.NumberColumn(width="small"),
                        "Percentage": st.column_config.ProgressColumn(
                            "Contribution", format="%.1f%%", min_value=0,
                            max_value=float(table_df["Percentage"].max()),
                        ),
                    },
                )

        # ── Recommendations ──────────────────────────────────────────────
        with tab4:
            kg = weight * LBS_TO_KG

            if prediction >= HEALTHY_THRESHOLD:
                status = (HIGH_C, "Body fat above the healthy range",
                          "Aim for a steady deficit rather than an aggressive cut. Slow progress is the kind that lasts.")
                recs = [
                    ("Cardio",
                     "3-4 sessions a week, 30-45 minutes each, at a pace where you can still hold a conversation. "
                     "Skip HIIT until you have a base: it drives hunger up more than it burns fat."),
                    ("Calorie deficit",
                     "Eat 300-500 kcal below your TDEE. That works out to 0.5-1 lb a week, fast enough to see and "
                     "slow enough to keep muscle. Track for two weeks before adjusting."),
                    ("Resistance training",
                     "Lift 2-3 times a week using compound movements: squat, hinge, press, pull. Consistency beats "
                     "load. Muscle you keep during a cut holds your metabolism up."),
                    ("Protein",
                     f"Eat {round(kg * 1.8)}-{round(kg * 2.2)} g of protein a day, spread across meals. It is the "
                     "single biggest lever for holding on to muscle in a deficit."),
                    ("Sleep",
                     "7-9 hours. Short sleep raises ghrelin, the hunger hormone, and erodes willpower. Fix sleep "
                     "before optimising anything else."),
                    ("Track progress",
                     "Weigh in at the same time each morning and look at the weekly average, since water alone "
                     "moves the scale 1-3 lbs. Re-measure your waist every two weeks."),
                ]
            elif prediction >= LEAN_THRESHOLD:
                status = (HEALTHY_C, "Healthy body fat range",
                          "You are in a good place. These focus on staying here and improving performance.")
                recs = [
                    ("Balanced nutrition",
                     "No deficit needed. Eat at maintenance and focus on quality: lean protein, complex carbs and "
                     "healthy fats at most meals."),
                    ("Strength training",
                     "3-4 sessions a week with progressive overload, adding a little weight or a few reps each week. "
                     "More muscle improves composition without changing scale weight."),
                    ("Cardiovascular fitness",
                     "2-3 sessions a week. A 30 minute walk or ride counts. It keeps your heart healthy and speeds up "
                     "recovery between lifting days."),
                    ("Protein",
                     f"Keep protein at {round(kg * 1.6)}-{round(kg * 2.0)} g a day. Having a good protein source at "
                     "each meal is enough."),
                    ("Hydration",
                     "About 35 ml per kg of bodyweight a day, more on training days. Pale yellow urine is the easiest "
                     "real-time check."),
                    ("Regular check-ins",
                     "Weigh in every few weeks and measure monthly. The goal is to catch slow drift early, while it is "
                     "still easy to reverse."),
                ]
            else:
                status = (LEAN_C, "Lean body composition",
                          "Being lean comes with higher maintenance demands. Recovery becomes the priority.")
                recs = [
                    ("Eat enough",
                     "Chronic undereating at low body fat suppresses testosterone, disrupts sleep and stalls muscle "
                     "growth. Eat at or above TDEE on training days."),
                    ("High protein",
                     f"Aim for {round(kg * 2.0)}-{round(kg * 2.4)} g of protein a day. At this level of leanness, "
                     "muscle is more vulnerable to breakdown."),
                    ("Prioritise recovery",
                     "Muscle is built during rest. Take 1-2 full rest days a week and get 8+ hours of sleep when "
                     "training hard. Overtraining while lean leads to injury quickly."),
                    ("Manage stress",
                     "Chronically high cortisol breaks down muscle and stores fat around the midsection, even in lean "
                     "people. Breathing work or lighter training weeks both help."),
                    ("Hydration and electrolytes",
                     "Lean bodies carry less water buffer. If you sweat heavily, watch sodium, potassium and magnesium: "
                     "cramps and fatigue are often electrolytes, not fitness."),
                    ("Periodic blood work",
                     "Test every 6-12 months. Testosterone, thyroid (T3/T4), ferritin and vitamin D are the markers "
                     "most likely to drop at sustained low body fat."),
                ]

            s_color, s_title, s_desc = status
            html(f"""
            <div class="rec-status" style="border-color:{s_color}">
              <div class="t" style="color:{s_color}">{s_title}</div>
              <div class="d">{s_desc}</div>
            </div>
            """)
            items = "".join(f'<div class="rec"><h4>{t}</h4><p>{b}</p></div>' for t, b in recs)
            html(f'<div class="recs">{items}</div>')

        # ── Nutrition ────────────────────────────────────────────────────
        with tab5:
            st.markdown("#### Calorie targets")

            html(f"""
            <div class="stats">
              <div class="stat"><div class="stat-label">Resting (BMR)</div>
                <div class="stat-value num">{bmr:,.0f}</div><div class="stat-note">kcal per day at rest</div></div>
              <div class="stat"><div class="stat-label">Maintenance (TDEE)</div>
                <div class="stat-value num">{tdee:,.0f}</div><div class="stat-note">kcal per day with activity</div></div>
              <div class="stat"><div class="stat-label">Fat loss</div>
                <div class="stat-value num">{tdee - 400:,.0f}</div><div class="stat-note">400 kcal deficit</div></div>
              <div class="stat"><div class="stat-label">Muscle gain</div>
                <div class="stat-value num">{tdee + 250:,.0f}</div><div class="stat-note">250 kcal surplus</div></div>
            </div>
            """)

            st.markdown("#### Macro split at maintenance")

            weight_kg  = weight * LBS_TO_KG
            protein_g  = round(weight_kg * 2.0)
            fat_g      = round(tdee * 0.28 / 9)
            carb_g     = max(round((tdee - protein_g * 4 - fat_g * 9) / 4), 0)
            total_kcal = round(tdee)
            meals      = 4

            protein_kcal = protein_g * 4
            fat_kcal     = fat_g * 9
            carb_kcal    = carb_g * 4

            macro_data = [
                (ACCENT,    "Protein",       "Muscle repair and satiety",            protein_g, protein_kcal),
                ("#8b93a8", "Fat",           "Hormones and fat-soluble vitamins",    fat_g,     fat_kcal),
                ("#c9cfdd", "Carbohydrates", "Primary fuel and glycogen",            carb_g,    carb_kcal),
            ]

            left, right = st.columns([1, 1.35], gap="large")
            with left:
                macro_fig = go.Figure(go.Pie(
                    labels=[m[1] for m in macro_data],
                    values=[m[4] for m in macro_data],
                    customdata=[f"{m[3]} g" for m in macro_data],
                    hovertemplate="<b>%{label}</b><br>%{value} kcal, %{customdata}<extra></extra>",
                    hole=0.72, sort=False, direction="clockwise",
                    marker=dict(colors=[m[0] for m in macro_data], line=dict(color="#111316", width=3)),
                    textinfo="none",
                ))
                macro_fig.add_annotation(
                    text=f"<span style='font-size:28px;font-weight:600'>{total_kcal:,}</span>"
                         f"<br><span style='color:{MUTED};font-size:12px'>kcal per day</span>",
                    x=0.5, y=0.5, showarrow=False,
                )
                style_fig(macro_fig, 280, showlegend=False, margin=dict(t=8, b=8, l=8, r=8))
                show_chart(macro_fig)

            with right:
                rows = ""
                for color, name, role, grams, kcal in macro_data:
                    share = round(kcal / total_kcal * 100) if total_kcal else 0
                    rows += f"""
                    <div class="macro">
                      <div class="sw" style="background:{color}"></div>
                      <div><div class="n">{name}</div><div class="r">{role}</div></div>
                      <div class="g num">{grams} g</div>
                      <div class="s num">{share}% · {round(grams / meals)} g/meal</div>
                    </div>
                    """
                html(f"""
                <div class="macros">
                  {rows}
                  <div class="macro-total">
                    <div class="k">Daily total, over {meals} meals</div>
                    <div class="v num">{total_kcal:,} kcal</div>
                  </div>
                </div>
                """)

            st.caption("BMR uses the Harris-Benedict equation with an activity multiplier. "
                       "See a dietitian for a personalised plan.")

        # ── History ──────────────────────────────────────────────────────
        with tab6:
            st.markdown("#### This session")

            hist_df = pd.DataFrame(st.session_state.history)
            display_cols = ["Timestamp", "Body Fat %", "BMI", "WHR",
                            "BMR (kcal)", "TDEE (kcal)", "Health Score", "Category"]

            if len(hist_df) > 1:
                trend_fig = go.Figure(go.Scatter(
                    x=hist_df["Timestamp"], y=hist_df["Body Fat %"], mode="lines+markers",
                    line=dict(color=ACCENT, width=2), marker=dict(size=7, color=ACCENT),
                    hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
                ))
                style_fig(
                    trend_fig, 260, margin=dict(t=10, b=30, l=40, r=10),
                    xaxis=dict(showgrid=False, linecolor=LINE, tickfont=dict(family=FONT_MONO, size=11, color=MUTED)),
                    yaxis=dict(gridcolor=LINE, zeroline=False, ticksuffix="%",
                               tickfont=dict(family=FONT_MONO, size=11, color=MUTED)),
                )
                show_chart(trend_fig)

                @st.fragment
                def render_comparison(df):
                    metric = st.selectbox("Compare across estimates",
                                          ["Body Fat %", "BMI", "Health Score", "WHR"])
                    bar_fig = px.bar(df, x="Timestamp", y=metric)
                    bar_fig.update_traces(marker_color=ACCENT, marker_line_width=0,
                                          hovertemplate="%{x}<br>%{y}<extra></extra>")
                    style_fig(
                        bar_fig, 260, bargap=0.45, margin=dict(t=10, b=30, l=40, r=10),
                        xaxis=dict(title=None, showgrid=False, linecolor=LINE,
                                   tickfont=dict(family=FONT_MONO, size=11, color=MUTED)),
                        yaxis=dict(title=None, gridcolor=LINE, zeroline=False,
                                   tickfont=dict(family=FONT_MONO, size=11, color=MUTED)),
                    )
                    show_chart(bar_fig)

                render_comparison(hist_df)
            else:
                html('<p class="section-note">Run another estimate to see a trend.</p>')

            st.dataframe(
                hist_df[display_cols],
                hide_index=True,
                width="stretch",
                column_config={
                    "Body Fat %":   st.column_config.NumberColumn(format="%.1f%%"),
                    "BMI":          st.column_config.NumberColumn(format="%.1f"),
                    "WHR":          st.column_config.NumberColumn(format="%.2f"),
                    "BMR (kcal)":   st.column_config.NumberColumn(format="%d"),
                    "TDEE (kcal)":  st.column_config.NumberColumn(format="%d"),
                    "Health Score": st.column_config.NumberColumn(format="%d"),
                },
            )

            c_dl, c_clear, _ = st.columns([1, 1, 2])
            with c_dl:
                st.download_button("Download history (CSV)",
                                   hist_df.to_csv(index=False).encode("utf-8"),
                                   "body_fat_history.csv", "text/csv", width="stretch")
            with c_clear:
                if st.button("Clear history", width="stretch"):
                    st.session_state.history = []
                    st.rerun()


    html("""
    <div class="foot">For education and self-tracking only. Not a clinical assessment.</div>
    """)
