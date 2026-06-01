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
        'Job',
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

job = st.number_input(
    "Job (0-3)",
    0,
    3,
    1
)

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
        job,
        housing,
        credit_amount,
        duration
    ]])

    probability = model.predict_proba([[
        age,
        sex,
        job,
        housing,
        credit_amount,
        duration
    ]])

    risk_score = probability[0][1] * 100

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.markdown(
            f'''
            <div class='result-bad'>
            ❌ HIGH RISK CUSTOMER<br><br>
            Risk Probability: {risk_score:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.warning("Reasons for High Risk")

        if credit_amount > 1000000:
            st.write("• Credit amount is very high")

        if duration > 36:
            st.write("• Loan duration is too long")

        if age < 21:
            st.write("• Applicant age is low")

        if housing == 2:
            st.write("• Customer lives in rented house")

        st.info("Suggestion: Reduce loan amount or duration.")

    else:

        st.markdown(
            f'''
            <div class='result-good'>
            ✅ LOW RISK CUSTOMER<br><br>
            Approval Probability: {100-risk_score:.2f}%
            </div>
            ''',
            unsafe_allow_html=True
        )

        st.success("Customer profile looks financially stable.")