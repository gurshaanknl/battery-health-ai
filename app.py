import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Battery Health Checker", page_icon="🔋", layout="centered")

model = joblib.load('battery_model.pkl')

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');

* { font-family: 'Space Grotesk', sans-serif; }

.stApp {
    background: radial-gradient(circle at 20% 20%, #0f2027 0%, #0a0e14 45%, #050709 100%);
    color: #e6f1ff;
}

/* Title with gradient glow */
h1 {
    background: linear-gradient(90deg, #00ffa3, #00d4ff 50%, #00ffa3);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
    font-weight: 700 !important;
    font-size: 3rem !important;
    text-shadow: 0 0 40px rgba(0,255,163,0.25);
}
@keyframes shine {
    to { background-position: 200% center; }
}

/* Subtitle */
.stMarkdown p {
    color: #8fa3b3;
    font-size: 1.05rem;
}

/* Glass card wrapping the inputs */
div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,255,163,0.15);
    border-radius: 20px;
    padding: 10px 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

/* Number input styling */
div[data-testid="stNumberInput"] label {
    color: #00ffa3 !important;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.8rem !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(0,0,0,0.4) !important;
    color: #e6f1ff !important;
    border: 1px solid rgba(0,255,163,0.25) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 10px !important;
    transition: all 0.25s ease;
}
div[data-testid="stNumberInput"] input:focus {
    border: 1px solid #00ffa3 !important;
    box-shadow: 0 0 20px rgba(0,255,163,0.35) !important;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #00ffa3, #00d4ff);
    color: #05070a;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-size: 1rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    width: 100%;
    margin-top: 20px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(0,255,163,0.3);
}
.stButton button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 8px 30px rgba(0,255,163,0.5);
}
.stButton button:active { transform: translateY(0) scale(0.99); }

/* Result boxes */
div[data-testid="stAlert"] {
    border-radius: 16px !important;
    padding: 22px !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    border: none !important;
    margin-top: 20px !important;
    animation: fadeIn 0.5s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ---------- APP CONTENT ----------
st.title("🔋 Battery Health Checker")
st.write("Enter a battery's readings to check if it's still good or worn out.")

ambient_temp = st.number_input("Ambient Temperature (°C)", value=24.000000, format="%.6f")
voltage_avg = st.number_input("Average Voltage (V)", value=3.500000, format="%.6f")
current_avg = st.number_input("Average Current (A)", value=-1.800000, format="%.6f")
temperature_avg = st.number_input("Average Battery Temperature (°C)", value=32.000000, format="%.6f")

if st.button("Check Battery"):
    input_data = np.array([[ambient_temp, voltage_avg, current_avg, temperature_avg]])
    prediction = model.predict(input_data)[0]

    if prediction == 'good':
        st.success("✅ REUSE — Battery is still healthy")
    else:
        st.error("♻️ RECYCLE — Battery is worn out")