import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Page config
st.set_page_config("Linear Regression", layout = "centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()} </style>", unsafe_allow_html = True)

load_css("style.css")

st.markdown("""
        <div class = "card">
            <h1> Linear Regression </h1>
            <p> Predict <b> Tip Amount </b> from <b> Total Bill </b> using Linear Regression </p>
        </div>
""", unsafe_allow_html = True)

# Load Data

@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df = load_data()

# Dataset Preview
st.markdown('<div class = "card">', unsafe_allow_html = True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html = True)

# Data Preparation
x, y = df[["total_bill"]], df["tip"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Feature Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Predictions
y_pred = model.predict(x_test_scaled)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Metrics Display
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("R² Score", f"{r2:.2f}")
st.markdown('</div>', unsafe_allow_html=True)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Regression Visualization")

fig, ax = plt.subplots()
sns.scatterplot(x=x_test["total_bill"], y=y_test, ax=ax, label="Actual", color="orange")
sns.lineplot(
    x=x_test["total_bill"],
    y=y_pred,
    ax=ax,
    label="Predicted",
    color="blue"
)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip Amount")
ax.legend()
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# Prediction Section
# Prediction Section
st.markdown('<div class="card prediction-card">', unsafe_allow_html=True)
st.subheader("Tip Prediction")

bill_amount = st.number_input(
    "Total Bill Amount ($)",
    min_value=0.0,
    value=25.0,
    step=1.0
)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

if st.button("Predict Tip Amount"):
    bill_scaled = scaler.transform([[bill_amount]])
    predicted_tip = model.predict(bill_scaled)[0]

    st.markdown(
        f"""
        <div class="prediction-result">
            <span class="label">Estimated Tip</span>
            <span class="value">${predicted_tip:.2f}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
