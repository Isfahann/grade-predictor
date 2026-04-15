import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── 1. Generate synthetic dataset ──────────────────────────────────────────
np.random.seed(42)
n = 500

study_hours     = np.random.uniform(1, 10, n)
attendance      = np.random.uniform(50, 100, n)
prev_score      = np.random.uniform(40, 100, n)
assignments     = np.random.uniform(50, 100, n)
sleep_hours     = np.random.uniform(4, 9, n)

# Grade formula with some noise
grade = (
    0.30 * study_hours * 10 +
    0.25 * attendance * 0.8 +
    0.25 * prev_score * 0.7 +
    0.10 * assignments * 0.5 +
    0.10 * sleep_hours * 5 +
    np.random.normal(0, 3, n)
).clip(0, 100)

df = pd.DataFrame({
    "study_hours":  study_hours,
    "attendance":   attendance,
    "prev_score":   prev_score,
    "assignments":  assignments,
    "sleep_hours":  sleep_hours,
    "grade":        grade
})

# ── 2. Train model ─────────────────────────────────────────────────────────
X = df.drop("grade", axis=1)
y = df["grade"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae    = mean_absolute_error(y_test, y_pred)
r2     = r2_score(y_test, y_pred)

# ── 3. Streamlit UI ────────────────────────────────────────────────────────
st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓")

st.title("🎓 Student Grade Predictor")
st.markdown("Fill in your study habits below to predict your exam grade.")

st.sidebar.header("Model Performance")
st.sidebar.metric("R² Score",  f"{r2:.2f}")
st.sidebar.metric("Mean Absolute Error", f"{mae:.2f} pts")

st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    study_h  = st.slider("Study Hours per Day",    1.0, 10.0, 5.0, 0.5)
    attend   = st.slider("Attendance (%)",         50.0, 100.0, 80.0, 1.0)
    prev     = st.slider("Previous Exam Score",    40.0, 100.0, 70.0, 1.0)

with col2:
    assign   = st.slider("Assignment Completion (%)", 50.0, 100.0, 80.0, 1.0)
    sleep    = st.slider("Sleep Hours per Night",   4.0, 9.0, 7.0, 0.5)

if st.button("Predict My Grade", use_container_width=True):
    input_data = pd.DataFrame([[study_h, attend, prev, assign, sleep]],
                              columns=X.columns)
    prediction = model.predict(input_data)[0]
    prediction = float(np.clip(prediction, 0, 100))

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction >= 80:
        grade_letter, colour = "A", "🟢"
    elif prediction >= 65:
        grade_letter, colour = "B", "🟡"
    elif prediction >= 50:
        grade_letter, colour = "C", "🟠"
    else:
        grade_letter, colour = "D/F", "🔴"

    st.metric(label="Predicted Score", value=f"{prediction:.1f} / 100")
    st.success(f"{colour} Grade: **{grade_letter}**")

    st.markdown("### 💡 Suggestions")
    if study_h < 4:
        st.warning("📚 Try studying at least 4 hours a day.")
    if attend < 75:
        st.warning("🏫 Attendance below 75% significantly impacts grades.")
    if sleep < 6:
        st.warning("😴 Getting less than 6 hours of sleep affects performance.")
    if study_h >= 6 and attend >= 80:
        st.info("✅ Keep it up — your habits look solid!")

st.markdown("---")
st.caption("Built with Python · scikit-learn · Streamlit")