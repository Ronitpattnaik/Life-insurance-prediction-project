import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #0b1120;
}

h1 {
    color: white;
    text-align: center;
    font-size: 48px;
}

.stButton > button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    height: 50px;
    border-radius: 10px;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

.result-good {
    background-color: #14532d;
    padding: 18px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    text-align: center;
}

.result-bad {
    background-color: #7f1d1d;
    padding: 18px;
    border-radius: 10px;
    color: white;
    font-size: 22px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------

st.title("Life Insurance / Credit Risk Prediction")

st.write("Enter applicant information to predict risk")

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("german_credit_data.csv")

df = df.dropna()

# -----------------------------
# ENCODING
# -----------------------------

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Housing'] = le.fit_transform(df['Housing'])
df['Saving accounts'] = le.fit_transform(df['Saving accounts'])
df['Checking account'] = le.fit_transform(df['Checking account'])
df['Purpose'] = le.fit_transform(df['Purpose'])
df['Risk'] = le.fit_transform(df['Risk'])

# -----------------------------
# FEATURES
# -----------------------------

X = df[['Age',
        'Sex',
        'Housing',
        'Credit amount',
        'Duration']]

y = df['Risk']

# -----------------------------
# MODEL
# -----------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X, y)

# -----------------------------
# USER INPUT
# -----------------------------

age = st.number_input("Age", 18, 75, 30)

sex_text = st.selectbox(
    "Sex",
    ["male", "female"]
)

if sex_text == "male":
    sex = 1
else:
    sex = 0

income = st.number_input(
    "Monthly Income (₹)",
    min_value=5000,
    max_value=500000,
    value=30000,
    step=5000
)

st.write(f"Selected Income: ₹ {income:,}")

housing_text = st.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

if housing_text == "own":
    housing = 1
elif housing_text == "rent":
    housing = 2
else:
    housing = 0

credit_amount = st.number_input(
    "Credit Amount (₹)",
    min_value=1000,
    max_value=5000000,
    value=100000,
    step=1000
)

st.write(f"Selected Credit Amount: ₹ {credit_amount:,}")

duration = st.number_input(
    "Duration (months)",
    1,
    72,
    12
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Risk"):

    prediction = model.predict([[
        age,
        sex,
        housing,
        credit_amount,
        duration
    ]])

    probability = model.predict_proba([[
        age,
        sex,
        housing,
        credit_amount,
        duration
    ]])

    risk_score = probability[0][1] * 100

    high_risk_reason = []

    # Manual business logic

    if age > 65:
        high_risk_reason.append("High age")

    if income < 20000:
        high_risk_reason.append("Low monthly income")

    if credit_amount > 800000:
        high_risk_reason.append("Very high credit amount")

    if duration > 48:
        high_risk_reason.append("Long repayment duration")

    if housing == 2:
        high_risk_reason.append("Customer lives in rented house")

    # Final Decision

    if len(high_risk_reason) >= 2:

        st.markdown(
            f'''
            <div class='result-bad'>
            ❌ HIGH RISK CUSTOMER<br><br>
            Risk Score: {risk_score:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.subheader("Why Customer is High Risk?")

        for reason in high_risk_reason:
            st.write(f"• {reason}")

        st.warning(
            "Suggestion: Reduce loan amount or repayment duration."
        )

    else:

        st.markdown(
            f'''
            <div class='result-good'>
            ✅ LOW RISK CUSTOMER<br><br>
            Approval Score: {100-risk_score:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.success("Customer profile appears financially stable.")
