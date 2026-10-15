import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        -45deg,
        #F0FDFA,
        #CCFBF1,
        #FFFFFF,
        #D1FAE5
    );
    background-size: 400% 400%;
    animation: gradientMove 15s ease infinite;
}

@keyframes gradientMove {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

#linear-gradient(135deg, #F8FAFC 0%, #DBEAFE 100%)

# Custom styling for clinical cockpit UI
st.markdown("""
<style>
    .metric-card {
        background-color: #32e641;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .badge-alert {
        background-color: #d4f211;
        color: #f87171;
        border: 1px solid rgba(239, 68, 50, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-stable {
        background-color: #10b981;
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-warn {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_food_database():
    """Loads pred_food.csv nutritional database with fallback standard foods."""
    if os.path.exists("pred_food.csv"):
        df = pd.read_csv("pred_food.csv")
    else:
        # Fallback dictionary if file is temporarily missing
        data = {
            "Food Name": ["Apple", "Banana", "White Rice (1 Bowl)", "Roti / Chapati (2 pcs)", "Biryani (Chicken)", "Masala Chai with Sugar", "Samosa (1 pc)", "Moong Dal (1 Bowl)", "Salad with Greens", "Oatmeal"],
            "Glycemic Index": [39, 51, 73, 62, 75, 68, 70, 38, 15, 55],
            "Calories": [52, 96, 205, 140, 380, 120, 260, 150, 45, 160],
            "Carbohydrates": [14.0, 23.0, 45.0, 30.0, 52.0, 18.0, 24.0, 20.0, 5.0, 27.0],
            "Protein": [0.3, 1.1, 4.2, 5.0, 16.0, 3.0, 3.5, 9.0, 2.0, 5.5],
            "Fat": [0.2, 0.2, 0.4, 0.8, 14.0, 3.5, 17.0, 2.0, 0.5, 3.0],
            "Fiber Content": [2.4, 2.6, 0.6, 2.5, 1.8, 0.0, 1.2, 5.2, 3.5, 4.0],
            "Suitable for Diabetes": [1, 1, 0, 1, 0, 0, 0, 1, 1, 1]
        }
        df = pd.DataFrame(data)
    
    # Ensure column formatting
    df["Food Name"] = df["Food Name"].astype(str).str.strip()
    return df


@st.cache_data(show_spinner=False)
def load_diabetes_dataset():
    """Loads diabetes_prediction_dataset.csv or creates realistic subset."""
    if os.path.exists("diabetes_prediction_dataset.csv"):
        df = pd.read_csv("diabetes_prediction_dataset.csv")
    else:
        # Fallback synthetic cohort
        np.random.seed(42)
        n = 5000
        genders = np.random.choice(["Male", "Female"], size=n)
        ages = np.random.uniform(18, 80, size=n)
        hypertension = np.random.choice([0, 1], p=[0.85, 0.15], size=n)
        heart_disease = np.random.choice([0, 1], p=[0.92, 0.08], size=n)
        smoking = np.random.choice(["never", "current", "former", "No Info"], size=n)
        bmi = np.random.normal(27.0, 5.0, size=n).clip(15.0, 55.0)
        hba1c = np.random.normal(5.8, 1.1, size=n).clip(4.0, 12.0)
        glucose = (ages * 0.4 + bmi * 1.8 + hba1c * 12.0 + np.random.normal(0, 15, size=n)).clip(70, 300)
        diabetes = ((glucose > 140) | (hba1c > 6.5)).astype(int)
        df = pd.DataFrame({
            "gender": genders, "age": ages, "hypertension": hypertension,
            "heart_disease": heart_disease, "smoking_history": smoking,
            "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose,
            "diabetes": diabetes
        })
    return df


@st.cache_resource(show_spinner=False)
def train_stage1_models(df):
    """
    Trains two models on the patient profile:
    1. Classifier: Predicts chronic diabetic risk probability (0-100%).
    2. Regressor: Predicts expected resting/baseline blood glucose (mg/dL).
    """
    feature_cols = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi"]
    cat_cols = ["gender", "smoking_history"]
    num_cols = ["age", "hypertension", "heart_disease", "bmi"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )
    
    # 1. Chronic Risk Classifier
    clf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(max_iter=120, random_state=42))
    ])
    clf_pipeline.fit(df[feature_cols], df["diabetes"])
    
    # 2. Baseline Glucose Regressor
    reg_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(max_iter=120, random_state=42))
    ])
    reg_pipeline.fit(df[feature_cols], df["blood_glucose_level"])
    
    return clf_pipeline, reg_pipeline


# Load and train
df_food = load_food_database()
df_diab = load_diabetes_dataset()
stage1_clf, stage1_reg = train_stage1_models(df_diab)


st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
    <div style="background: linear-gradient(135deg, #F97316, #F43F5E); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
        🧬
    </div>
    <div>
        <h3 style="margin: 0; font-size: 1.15rem; color: #1729e8;">AuraTwin Core</h3>
        <p style="margin: 0; font-size: 0.75rem; color: #e817bb;">Two-Stage Metabolic Forecaster</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Stage 1: Chronic Patient Profile")
st.sidebar.caption("Baseline EHR & demographic factors (trained on 100k cohort)")

gender = st.sidebar.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
age = st.sidebar.slider("Age (Years)", min_value=18, max_value=85, value=48, step=1)

bmiknown = st.sidebar.checkbox("I know my BMI", value=False)
st.sidebar.caption("If unchecked, you can input height and weight to compute BMI automatically.")
st.sidebar.caption("BMI = Weight (kg) / (Height (m))²")
st.sidebar.caption("BMI meaning: <18.5 = Underweight, 18.5-24.9 = Normal, 25-29.9 = Overweight, 30+ = Obese")
if bmiknown:
    bmi = st.sidebar.slider("BMI (kg/m²)", min_value=15.0, max_value=55.0, value=27.0, step=0.1)

if not bmiknown:
    height = st.sidebar.slider("Height (cm)", min_value=140, max_value=210, value=170, step=1)
    st.sidebar.caption("Height in centimeters")
    weight = st.sidebar.slider("Weight (kg)", min_value=40, max_value=150, value=78, step=1)
    st.sidebar.caption("Weight in kilograms")

    bmi = weight / ((height / 100) ** 2)


col_med1, col_med2 = st.sidebar.columns(2)
with col_med1:
    hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    st.sidebar.caption("Hypertension means High blood pressure")
with col_med2:
    heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    

smoking_history = st.sidebar.selectbox("Smoking History", options=["never", "current", "former", "ever", "not current", "No Info"], index=0)

# Build feature row for ML inference
patient_profile_df = pd.DataFrame([{
    "gender": gender,
    "age": float(age),
    "hypertension": int(hypertension),
    "heart_disease": int(heart_disease),
    "smoking_history": smoking_history,
    "bmi": float(bmi)
}])


stage1_prob = float(stage1_clf.predict_proba(patient_profile_df)[0][1]) * 100
stage1_baseline_glucose = float(stage1_reg.predict(patient_profile_df)[0])

# Effective insulin resistance modifier derived from profile
# Higher BMI, age, and chronic risk yield higher postprandial sensitivity deficit
insulin_resistance_factor = 1.0 + (stage1_prob / 100.0) * 0.65 + max(0, (bmi - 24.0) * 0.02)


st.title("AuraTwin: Two-Stage Type 2 Diabetes Digital Twin")
st.caption("""
🧬 **How AuraTwin Works**

1. 🩺 **Build Your Profile**  
Enter your health metrics and lifestyle details to establish your personalized metabolic baseline.

2. 🍽️ **Simulate Your Meal**  
Add foods and quantities to estimate meal nutrition, glycemic index (GI), and glycemic load (GL).

3. 📊 **Explore Your Response**  
Our model combines your profile and meal data to estimate your potential glucose response and visualize personalized insights.

*AuraTwin is a prototype for educational and wellness exploration, not a clinically validated diagnostic tool.*
""")

col_kpi1, col_kpi2, col_kpi3 = st.columns(3)

with col_kpi1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>STAGE 1: CHRONIC RISK PROBABILITY</span>", unsafe_allow_html=True)
    risk_color = "#ef4444" if stage1_prob > 50 else ("#f59e0b" if stage1_prob > 25 else "#10b981")
    st.markdown(f"<h2 style='color: {risk_color}; margin: 4px 0;'>{stage1_prob:.1f}%</h2>", unsafe_allow_html=True)
    st.caption("Trained on 100k patient records via HistGradientBoosting")
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>STAGE 1: BASELINE RESTING GLUCOSE</span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: #38bdf8; margin: 4px 0;'>{stage1_baseline_glucose:.1f} <span style='font-size: 1rem; color: #94a3b8;'>mg/dL</span></h2>", unsafe_allow_html=True)
    st.caption("Estimated fasting / pre-meal steady state")
    st.markdown('</div>', unsafe_allow_html=True)

with col_kpi3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(f"<span style='color: #94a3b8; font-size: 0.8rem;'>INSULIN RESISTANCE MULTIPLIER (S_I)</span>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: #a78bfa; margin: 4px 0;'>{insulin_resistance_factor:.2f}x</h2>", unsafe_allow_html=True)
    st.caption("Excursion susceptibility computed from clinical profile")
    st.markdown('</div>', unsafe_allow_html=True)


st.markdown("---")

# --- Stage 2: Meal Builder ---
st.markdown("### 🍲 Build Your Meal")
st.caption(
    "Search for foods, add them to your meal, and set the quantity "
    "of each item independently."
)

available_foods = sorted(
    df_food["Food Name"].dropna().astype(str).unique().tolist()
)

# Keep the meal selections across Streamlit reruns
if "meal_quantities" not in st.session_state:
    st.session_state.meal_quantities = {}

# Searchable food selector (Streamlit selectbox supports typing to search)
food_to_add = st.selectbox(
    "🔍 Search and select a food",
    options=[""] + [
        food for food in available_foods
        if food not in st.session_state.meal_quantities
    ],
    format_func=lambda x: "Choose a food..." if x == "" else x,
    key="food_to_add",
)

if st.button("➕ Add food", disabled=(food_to_add == "")):
    st.session_state.meal_quantities.setdefault(food_to_add, 1.0)
    st.rerun()

st.markdown("#### Your meal")

if not st.session_state.meal_quantities:
    st.info("No food added yet. Search above to start building your meal.")
else:
    for food in list(st.session_state.meal_quantities.keys()):
        col_name, col_qty, col_remove = st.columns([3, 2, 1])

        with col_name:
            st.write(f"🍽️ **{food}**")

        with col_qty:
            quantity = st.number_input(
                "Quantity (servings)",
                min_value=0.25,
                max_value=20.0,
                value=float(st.session_state.meal_quantities[food]),
                step=0.25,
                key=f"qty_{food}",
            )
            st.session_state.meal_quantities[food] = quantity

        with col_remove:
            st.write("")  # Align button with the quantity input
            if st.button("🗑️", key=f"remove_{food}", help=f"Remove {food}"):
                del st.session_state.meal_quantities[food]
                st.rerun()

# --- Calculate nutrition using each food's own quantity ---
food_rows = []

for food, quantity in st.session_state.meal_quantities.items():
    matching = df_food[df_food["Food Name"] == food]

    if matching.empty:
        continue

    item = matching.iloc[0]
    food_rows.append({
        "Food": food,
        "Quantity": quantity,
        "GI": float(item["Glycemic Index"]),
        "Carbs": float(item["Carbohydrates"]) * quantity,
        "Calories": float(item["Calories"]) * quantity,
        "Fiber": float(item["Fiber Content"]) * quantity,
        "Suitable": float(item["Suitable for Diabetes"]),
    })

meal_df = pd.DataFrame(food_rows)

if not meal_df.empty:
    total_carbs = float(meal_df["Carbs"].sum())
    total_calories = float(meal_df["Calories"].sum())
    total_fiber = float(meal_df["Fiber"].sum())

    # Estimated meal GI, weighted by each food's carbohydrate contribution
    if total_carbs > 0:
        avg_gi = float(
            (meal_df["GI"] * meal_df["Carbs"]).sum() / total_carbs
        )
    else:
        avg_gi = 0.0

    # Total glycemic load accounts for both GI and quantity
    glycemic_load = float(
        (meal_df["GI"] * meal_df["Carbs"]).sum() / 100.0
    )

    suitable_diab_pct = float(meal_df["Suitable"].mean() * 100)

    st.dataframe(
        meal_df[["Food", "Quantity", "Carbs", "Calories", "Fiber", "GI"]],
        hide_index=True,
        use_container_width=True,
    )
else:
    total_carbs = 0.0
    total_calories = 0.0
    total_fiber = 0.0
    avg_gi = 0.0
    glycemic_load = 0.0
    suitable_diab_pct = 0.0

# --- Exercise and activity inputs ---
col_act1, col_act2, col_act3 = st.columns(3)

with col_act1:
    exercise_mins = st.slider(
        "🏃 Exercise / Walk Done in Last Hour (mins)",
        min_value=0, max_value=60, value=10, step=5,
    )

with col_act2:
    exercise_intensity = st.selectbox(
        "Exercise Intensity",
        options=["Light Walk", "Brisk Walk / Jog", "Intense Workout"],
        index=0,
    )

with col_act3:
    water_intake = st.selectbox(
        "Hydration / Water Intake",
        options=["Adequate (>500ml)", "Low / Dehydrated"],
        index=0,
    )


# Intensity factor for exercise clearance
intensity_factor = 0.4 if exercise_intensity == "Light Walk" else (0.8 if exercise_intensity == "Brisk Walk / Jog" else 1.2)
exercise_clearance = exercise_mins * intensity_factor

# Hydration attenuation
hydration_buff = 0.92 if water_intake == "Adequate (>500ml)" else 1.05

# Acute Glycemic Surge formulation (modified Minimal Model approximation):
# Net Surge = [ (GL * 1.55) - (Fiber * 1.8) ] * Insulin Resistance Factor * Hydration - Exercise Clearance
raw_surge = ((glycemic_load * 1.55) - (total_fiber * 1.8)) * insulin_resistance_factor * hydration_buff
net_acute_surge = max(0.0, raw_surge - exercise_clearance)

# Final Projected Real-Time Blood Glucose
projected_peak_glucose = round(stage1_baseline_glucose + net_acute_surge, 1)


with st.expander("🔎 View Ingested Nutritional Breakdown (Stage 2 Lookup)", expanded=True):
    subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
    subcol1.metric("Total Carbs Ingested", f"{total_carbs:.1f} g")
    subcol2.metric("Average Glycemic Index", f"{avg_gi:.1f}")
    subcol3.metric("Total Glycemic Load (GL)", f"{glycemic_load:.1f}")
    subcol4.metric("Dietary Fiber Buffer", f"{total_fiber:.1f} g")
    subcol5.metric("Exercise Clearance", f"-{exercise_clearance:.1f} mg/dL")


st.markdown("---")
st.subheader("🎯 Stage 3: The Combined Clinical Verdict")

verdict_col1, verdict_col2 = st.columns([1, 2])

with verdict_col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("<span style='color: #94a3b8; font-size: 0.85rem;'>PROJECTED POSTPRANDIAL GLUCOSE</span>", unsafe_allow_html=True)
    
    # Classify clinical zone
    if projected_peak_glucose > 180:
        badge_html = '<span class="badge-alert">🚨 ACUTE HYPERGLYCEMIC SPIKE</span>'
        color_val = "#ef4444"
    elif projected_peak_glucose > 140:
        badge_html = '<span class="badge-warn">⚠️ ELEVATED POSTPRANDIAL EXCURSION</span>'
        color_val = "#f59e0b"
    else:
        badge_html = '<span class="badge-stable">✅ NORMAL GLYCEMIC REGULATION</span>'
        color_val = "#10b981"

    st.markdown(f"<h1 style='color: {color_val}; font-size: 2.8rem; margin: 8px 0;'>{projected_peak_glucose} <span style='font-size: 1.1rem; color: #94a3b8;'>mg/dL</span></h1>", unsafe_allow_html=True)
    st.markdown(badge_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    - **Stage 1 Baseline:** `{stage1_baseline_glucose:.1f} mg/dL`
    - **Stage 2 Acute Surge:** `+{net_acute_surge:.1f} mg/dL`
    - **Chronic Profile Vulnerability:** `{insulin_resistance_factor:.2f}x multiplier`
    """)
    st.markdown('</div>', unsafe_allow_html=True)



with verdict_col2:
    st.markdown("#### 🧪 Proactive What-If Counterfactual Sandbox")
    st.caption("Explore how lifestyle changes affect the model's projected peak.")

    cf_walk = st.slider(
        "Add Immediate Brisk Walking (Minutes)",
        min_value=0,
        max_value=45,
        value=20,
        step=5
    )

    cf_fiber = st.checkbox(
        "Simulate Pre-meal Soluble Fiber Preload (+5g Psyllium)",
        value=False
    )

    # Illustrative model assumptions; validate before clinical use
    walk_effect_per_min = 0.95
    fiber_attenuation = 0.82 if cf_fiber else 1.0

    # Start from the original model's predicted glucose excursion
    baseline_peak = projected_peak_glucose
    baseline_excursion = max(
        0.0, baseline_peak - stage1_baseline_glucose
    )

    # Apply interventions consistently to the excursion
    walk_reduction = min(
        baseline_excursion,
        cf_walk * walk_effect_per_min
    )

    adjusted_excursion = max(
        0.0,
        baseline_excursion * fiber_attenuation - walk_reduction
    )

    counterfactual_peak = (
        stage1_baseline_glucose + adjusted_excursion
    )

    reduction = baseline_peak - counterfactual_peak

    st.info(
        f"💡 **Simulated Outcome**\n\n"
        f"- Original projected peak: **{baseline_peak:.1f} mg/dL**\n"
        f"- Projected peak after intervention: **{counterfactual_peak:.1f} mg/dL**\n"
        f"- Difference from baseline: **{reduction:.1f} mg/dL**"
    )


st.markdown("#### 📈 Glycemic Trajectory: Baseline vs Acute Disturbance vs Intervention")

time_steps = [0, 15, 30, 45, 60, 90, 120]  # Minutes post-meal

# Generate natural curve trajectory
natural_curve = []
counterfactual_curve = []

for t in time_steps:
    if t == 0:
        val_nat = stage1_baseline_glucose
        val_cf = stage1_baseline_glucose
    elif t <= 60:
        # Rising postprandial surge
        progress = math.sin((t / 60) * (math.pi / 2))
        val_nat = stage1_baseline_glucose + (net_acute_surge * progress)
        val_cf = stage1_baseline_glucose + ((counterfactual_peak - stage1_baseline_glucose) * progress)
    else:
        # Clearance phase back toward baseline
        decay_factor = math.exp(-(t - 60) / 40.0)
        val_nat = stage1_baseline_glucose + (net_acute_surge * decay_factor)
        val_cf = stage1_baseline_glucose + ((counterfactual_peak - stage1_baseline_glucose) * decay_factor)
        
    natural_curve.append(round(val_nat, 1))
    counterfactual_curve.append(round(val_cf, 1))

fig = go.Figure()

# Normal Target Band (70 - 140 mg/dL)
fig.add_hrect(
    y0=70, y1=140,
    fillcolor="rgba(16, 185, 129, 0.08)",
    layer="below", line_width=0,
    annotation_text="Optimal Target Range (70-140 mg/dL)",
    annotation_position="top left",
    annotation_font=dict(color="#10b981", size=10)
)

# Hyperglycemia Risk Threshold line (180 mg/dL)
fig.add_hline(
    y=180, line_dash="dot", line_color="#ef4444",
    annotation_text="Hyperglycemic Threshold (>180 mg/dL)",
    annotation_position="top right",
    annotation_font=dict(color="#ef4444", size=10)
)

# Natural Trajectory
fig.add_trace(go.Scatter(
    x=[f"+{t}m" for t in time_steps],
    y=natural_curve,
    mode="lines+markers",
    name="<span style='color: #f59e0b;'>Natural Projected Surge</span>",
    line=dict(color="#f59e0b", width=3),
    marker=dict(size=6)
))

# Counterfactual Intervened Trajectory
fig.add_trace(go.Scatter(
    x=[f"+{t}m" for t in time_steps],
    y=counterfactual_curve,
    mode="lines+markers",
    name="<span style='color: #10b981;'>Intervened Trajectory (Walk/Fiber)</span>",
    line=dict(color="#10b981", width=3, dash="dash"),
    marker=dict(size=6)
))

fig.update_layout(
    template="plotly_dark",
    plot_bgcolor="#090d16",
    paper_bgcolor="#090d16",
    height=380,
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis_title="Time Since Meal Ingestion",
    yaxis_title="Blood Glucose (mg/dL)",
    yaxis=dict(range=[60, max(240, projected_peak_glucose + 30)]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

st.text('''                                                                                                                                                                                    




Disclaimer: AuraTwin is a prototype for educational and wellness exploration. It is not a clinically validated diagnostic tool. Always consult healthcare professionals for medical advice. Predictions are based on statistical models and may not reflect individual responses. Use responsibly.''')
st.text("© 2026 AuraTwin. All rights reserved.")
st.text("Made with ❤️ by Abir for research and educational purposes only.")