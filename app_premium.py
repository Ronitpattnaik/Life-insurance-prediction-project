import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AI Risk Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS & STYLING
# ============================================================================

custom_css = """
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        overflow-x: hidden;
    }
    
    .main {
        background: transparent;
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1a3e 100%);
        border-right: 2px solid rgba(100, 200, 255, 0.2);
    }
    
    /* Main Title Styling */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
        text-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
    }
    
    h2 {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #00d4ff;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* Card Styling */
    .glass-card {
        background: rgba(20, 33, 61, 0.8);
        border: 1px solid rgba(100, 200, 255, 0.3);
        border-radius: 15px;
        padding: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(100, 200, 255, 0.6);
        box-shadow: 0 12px 48px rgba(0, 212, 255, 0.2);
        transform: translateY(-5px);
    }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        border: 1px solid rgba(100, 200, 255, 0.4);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(100, 200, 255, 0.8);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        transform: scale(1.05);
    }
    
    .metric-card h4 {
        color: rgba(200, 210, 255, 0.8);
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    
    .metric-card .value {
        color: #00d4ff;
        font-size: 2rem;
        font-weight: 800;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 12px 32px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: rgba(30, 50, 90, 0.9) !important;
        border: 1px solid rgba(100, 200, 255, 0.3) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        padding: 10px !important;
        font-size: 0.95rem !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #7c3aed 100%);
    }
    
    /* Alert Styling */
    .stAlert {
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 1px solid rgba(16, 185, 129, 0.5) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.2) !important;
        border: 1px solid rgba(245, 158, 11, 0.5) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
    }
    
    /* Text Colors */
    .success-text {
        color: #10b981;
        font-weight: 600;
    }
    
    .warning-text {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .error-text {
        color: #ef4444;
        font-weight: 600;
    }
    
    .info-text {
        color: #00d4ff;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px 8px 0 0;
        background: rgba(100, 200, 255, 0.1);
        border: 1px solid rgba(100, 200, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3) 0%, rgba(124, 58, 237, 0.3) 100%);
        border: 1px solid rgba(100, 200, 255, 0.5);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(100, 200, 255, 0.1);
        border-radius: 8px;
    }
    
    /* Data Table */
    .stDataFrame {
        background: rgba(20, 33, 61, 0.6) !important;
    }
    
    /* Divider */
    .stDivider {
        border-color: rgba(100, 200, 255, 0.2);
    }
    
    /* Animation */
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        50% {
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.6);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .glow-card {
        animation: glow 2s ease-in-out infinite;
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    .slide-in {
        animation: slideInLeft 0.5s ease-out;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'scaler' not in st.session_state:
    st.session_state.scaler = None

if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}

if 'df' not in st.session_state:
    st.session_state.df = None

if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('german_credit_data.csv', index_col=0)
    df = df.dropna()
    return df

@st.cache_resource
def train_model_func(df):
    """Train the logistic regression model"""
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Risk']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Features and target
    X = df_processed[['Age', 'Sex', 'Housing', 'Credit amount', 'Duration']]
    y = df_processed['Risk']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
    
    return model, scaler, label_encoders, accuracy, X_test_scaled, y_test

def get_model_metrics(model, X_test, y_test):
    """Calculate model metrics"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc_score': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics

def predict_risk(age, sex, housing, credit_amount, duration):
    """Predict risk for new data"""
    model = st.session_state.model
    scaler = st.session_state.scaler
    label_encoders = st.session_state.label_encoders
    
    # Encode inputs
    sex_encoded = label_encoders['Sex'].transform([sex])[0]
    housing_encoded = label_encoders['Housing'].transform([housing])[0]
    
    # Prepare features
    features = np.array([[age, sex_encoded, housing_encoded, credit_amount, duration]])
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability

def generate_recommendations(age, income, credit_amount, duration, housing, prediction):
    """Generate intelligent recommendations"""
    recommendations = []
    risk_score = 0
    
    if age > 65:
        recommendations.append("🎯 Consider a younger applicant or provide additional collateral")
        risk_score += 2
    
    if income < 20000:
        recommendations.append("💰 Applicant should improve monthly income before applying")
        risk_score += 2
    
    if credit_amount > 800000:
        recommendations.append("📉 Reduce the credit amount requested")
        risk_score += 2
    
    if duration > 48:
        recommendations.append("⏱️ Reduce the loan duration to lower monthly obligations")
        risk_score += 2
    
    if housing == "rent":
        recommendations.append("🏠 Consider requiring home ownership or additional security")
        risk_score += 1
    
    if risk_score >= 4 and prediction == 1:
        recommendations.append("⚠️ High-risk profile detected - Recommend rejection or guarantor")
    elif prediction == 1:
        recommendations.append("🤝 Consider approval with additional conditions")
    else:
        recommendations.append("✅ Applicant is financially stable - Safe to approve")
    
    return recommendations

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_metric_card(label, value, icon="📊", color="#00d4ff"):
    """Create an animated metric card"""
    return f"""
    <div class="metric-card fade-in">
        <div style="font-size: 1.5rem; margin-bottom: 5px;">{icon}</div>
        <h4>{label}</h4>
        <div class="value" style="color: {color};">{value}</div>
    </div>
    """

def plot_age_distribution(df):
    """Age distribution chart"""
    fig = px.histogram(
        df, x='Age', nbins=30,
        title='Age Distribution of Customers',
        color_discrete_sequence=['#00d4ff'],
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12),
        hovermode='x unified'
    )
    return fig

def plot_credit_analysis(df):
    """Credit amount analysis"""
    fig = px.box(
        df, y='Credit amount',
        title='Credit Amount Distribution',
        color_discrete_sequence=['#7c3aed'],
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12)
    )
    return fig

def plot_risk_comparison(df):
    """Risk comparison chart"""
    risk_counts = df['Risk'].value_counts()
    fig = go.Figure(data=[
        go.Bar(
            x=['Bad Risk', 'Good Risk'],
            y=risk_counts.values,
            marker=dict(
                color=['#ef4444', '#10b981'],
                line=dict(color=['#dc2626', '#059669'], width=2)
            )
        )
    ])
    fig.update_layout(
        title='Good vs Bad Risk Customers',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12),
        showlegend=False,
        template='plotly_dark'
    )
    return fig

def plot_housing_risk(df):
    """Housing type vs risk"""
    housing_risk = pd.crosstab(df['Housing'], df['Risk'])
    fig = px.bar(
        housing_risk,
        barmode='group',
        title='Housing Type vs Risk Profile',
        color_discrete_sequence=['#ef4444', '#10b981'],
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12)
    )
    return fig

def plot_correlation_heatmap(df):
    """Correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Feature Correlation Matrix',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12),
        template='plotly_dark'
    )
    return fig

def plot_duration_analysis(df):
    """Loan duration analysis"""
    fig = px.histogram(
        df, x='Duration',
        title='Loan Duration Distribution',
        nbins=25,
        color_discrete_sequence=['#7c3aed'],
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12)
    )
    return fig

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Good', 'Predicted Bad'],
        y=['Actual Good', 'Actual Bad'],
        colorscale='RdYlGn',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 14},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(20,33,61,0.6)',
        font=dict(color='white', size=12),
        template='plotly_dark'
    )
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load data and train model
    if st.session_state.df is None:
        st.session_state.df = load_data()
        model, scaler, encoders, accuracy, X_test, y_test = train_model_func(st.session_state.df)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.label_encoders = encoders
        st.session_state.accuracy = accuracy
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
    
    df = st.session_state.df
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🏦 AI Risk Prediction")
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "🔮 Prediction", "📊 Analytics", "📈 Dataset", "⏱️ History", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Info
        st.markdown("### 📚 Model Information")
        st.info(f"""
        **Model Type:** Logistic Regression
        **Accuracy:** {st.session_state.accuracy:.2%}
        **Total Samples:** {len(df)}
        **Features:** 5
        """)
        
        st.markdown("### ⏰ Current Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Date", datetime.now().strftime("%Y-%m-%d"))
        with col2:
            st.metric("Time", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer Info")
        st.caption("**Developed by:** Ronit Pattnaik")
        st.caption("**ML Framework:** Scikit-learn + Streamlit")
        st.caption("**Dataset:** German Credit Data")
    
    # Main pages
    if page == "🏠 Dashboard":
        show_dashboard(df)
    elif page == "🔮 Prediction":
        show_prediction()
    elif page == "📊 Analytics":
        show_analytics(df)
    elif page == "📈 Dataset":
        show_dataset(df)
    elif page == "⏱️ History":
        show_history()
    elif page == "ℹ️ About":
        show_about()

def show_dashboard(df):
    """Dashboard Page"""
    st.markdown("<h1>🏦 AI Based Life Insurance Risk Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; color: rgba(200,210,255,0.8); text-align: center; margin-bottom: 2rem;'>Predict customer financial risk using advanced Machine Learning</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistics Cards
    st.markdown("### 📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(create_metric_card(
            "Total Customers",
            f"{len(df):,}",
            "👥",
            "#00d4ff"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Avg Credit Amount",
            f"₹{df['Credit amount'].mean():,.0f}",
            "💰",
            "#7c3aed"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "Model Accuracy",
            f"{st.session_state.accuracy:.2%}",
            "🎯",
            "#10b981"
        ), unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    with col4:
        bad_count = (df['Risk'] == 'bad').sum()
        st.markdown(create_metric_card(
            "High Risk Customers",
            f"{bad_count:,}",
            "⚠️",
            "#ef4444"
        ), unsafe_allow_html=True)
    
    with col5:
        good_count = (df['Risk'] == 'good').sum()
        st.markdown(create_metric_card(
            "Low Risk Customers",
            f"{good_count:,}",
            "✅",
            "#10b981"
        ), unsafe_allow_html=True)
    
    with col6:
        st.markdown(create_metric_card(
            "Avg Loan Duration",
            f"{df['Duration'].mean():.0f} Months",
            "⏳",
            "#f59e0b"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Insights
    st.markdown("### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>📈 Dataset Overview</h3>
            <p><strong>Total Records:</strong> """ + f"{len(df):,}" + """</p>
            <p><strong>Age Range:</strong> """ + f"{df['Age'].min()}-{df['Age'].max()} years" + """</p>
            <p><strong>Credit Range:</strong> ₹""" + f"{df['Credit amount'].min():,.0f} - ₹{df['Credit amount'].max():,.0f}" + """</p>
            <p><strong>Avg Age:</strong> """ + f"{df['Age'].mean():.1f} years" + """</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        good_pct = (good_count / len(df)) * 100
        bad_pct = (bad_count / len(df)) * 100
        st.markdown(f"""
        <div class="glass-card">
            <h3>🎯 Risk Distribution</h3>
            <p><strong>Good Risk:</strong> {good_count:,} ({good_pct:.1f}%)</p>
            <p><strong>Bad Risk:</strong> {bad_count:,} ({bad_pct:.1f}%)</p>
            <p><strong>Risk Ratio:</strong> {(bad_pct/good_pct):.2f}:1</p>
            <p><strong>Safe Approval Rate:</strong> {good_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Statistics
    st.markdown("### 📊 Feature Statistics")
    
    stats_df = pd.DataFrame({
        'Feature': ['Age', 'Credit Amount', 'Duration'],
        'Min': [
            f"{df['Age'].min()}",
            f"₹{df['Credit amount'].min():,}",
            f"{df['Duration'].min()} mo"
        ],
        'Max': [
            f"{df['Age'].max()}",
            f"₹{df['Credit amount'].max():,}",
            f"{df['Duration'].max()} mo"
        ],
        'Mean': [
            f"{df['Age'].mean():.1f}",
            f"₹{df['Credit amount'].mean():,.0f}",
            f"{df['Duration'].mean():.1f} mo"
        ],
        'Std Dev': [
            f"{df['Age'].std():.1f}",
            f"₹{df['Credit amount'].std():,.0f}",
            f"{df['Duration'].std():.1f}"
        ]
    })
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

def show_prediction():
    """Prediction Page"""
    st.markdown("<h1>🔮 Risk Prediction Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(200,210,255,0.8); margin-bottom: 2rem;'>Enter applicant information to predict financial risk</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("👤 Age", 18, 75, 30, help="Applicant's age in years")
        
        with col2:
            sex = st.selectbox("👥 Gender", ["male", "female"], help="Applicant's gender")
        
        with col3:
            housing = st.selectbox("🏠 Housing Type", ["own", "rent", "free"], help="Current housing arrangement")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            income = st.number_input(
                "💵 Monthly Income (₹)",
                min_value=5000,
                max_value=500000,
                value=30000,
                step=5000,
                help="Monthly income of applicant"
            )
        
        with col5:
            credit_amount = st.number_input(
                "💰 Credit Amount (₹)",
                min_value=1000,
                max_value=5000000,
                value=100000,
                step=10000,
                help="Requested credit/loan amount"
            )
        
        with col6:
            duration = st.slider(
                "⏱️ Duration (Months)",
                1, 72, 12,
                help="Loan repayment duration"
            )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🎯 PREDICT RISK", use_container_width=True):
            with st.spinner("🔄 Analyzing risk profile..."):
                prediction, probability = predict_risk(age, sex, housing, credit_amount, duration)
                
                # Store in history
                history_entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'age': age,
                    'sex': sex,
                    'housing': housing,
                    'income': income,
                    'credit': credit_amount,
                    'duration': duration,
                    'prediction': 'Bad Risk' if prediction == 1 else 'Good Risk',
                    'good_prob': probability[0],
                    'bad_prob': probability[1]
                }
                st.session_state.prediction_history.append(history_entry)
                
                st.markdown("---")
                
                # Display Results
                if prediction == 0:
                    st.markdown("""
                    <div class="glass-card fade-in" style="border-color: rgba(16, 185, 129, 0.6); background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%); text-align: center; padding: 40px;">
                        <div style="font-size: 3rem; margin-bottom: 15px;">✅</div>
                        <h2 style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">LOW RISK CUSTOMER</h2>
                        <div style="margin: 20px 0; font-size: 1.5rem; color: #10b981; font-weight: 800;">
                    """ + f"{probability[0]*100:.1f}% Approval Probability" + """
                        </div>
                        <p style="color: rgba(200, 210, 255, 0.9); font-size: 1.1rem; margin-top: 15px;">Customer profile appears financially stable ✨</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="glass-card fade-in" style="border-color: rgba(239, 68, 68, 0.6); background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%); text-align: center; padding: 40px;">
                        <div style="font-size: 3rem; margin-bottom: 15px;">⚠️</div>
                        <h2 style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">HIGH RISK CUSTOMER</h2>
                        <div style="margin: 20px 0; font-size: 1.5rem; color: #ef4444; font-weight: 800;">
                    {probability[1]*100:.1f}% Risk Score
                        </div>
                        <p style="color: rgba(200, 210, 255, 0.9); font-size: 1.1rem; margin-top: 15px;">Additional scrutiny recommended ⚡</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Risk Factors
                st.markdown("### 🔍 Risk Analysis")
                recommendations = generate_recommendations(age, income, credit_amount, duration, housing, prediction)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Risk Factors")
                    for i, rec in enumerate(recommendations[:3], 1):
                        if "Safe" in rec or "✅" in rec or "approve" in rec:
                            st.success(rec)
                        elif "High-risk" in rec or "⚠️" in rec or "rejection" in rec:
                            st.error(rec)
                        else:
                            st.info(rec)
                
                with col2:
                    st.markdown("### 💡 Recommendations")
                    for rec in recommendations[3:]:
                        if "Safe" in rec or "✅" in rec or "approve" in rec:
                            st.success(rec)
                        elif "High-risk" in rec or "⚠️" in rec or "rejection" in rec:
                            st.error(rec)
                        else:
                            st.warning(rec)
                
                st.markdown("---")
                
                # Probability Visualization
                st.markdown("### 📊 Prediction Confidence")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['Good Risk', 'Bad Risk'],
                        y=[probability[0]*100, probability[1]*100],
                        marker=dict(
                            color=['#10b981', '#ef4444'],
                            line=dict(color=['#059669', '#dc2626'], width=2)
                        ),
                        text=[f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Risk Probability Distribution",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(20,33,61,0.6)',
                    font=dict(color='white', size=12),
                    showlegend=False,
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def show_analytics(df):
    """Analytics Page"""
    st.markdown("<h1>📊 Advanced Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(200,210,255,0.8); margin-bottom: 2rem;'>Comprehensive data analysis and insights</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Performance
    st.markdown("### 🤖 Model Performance Metrics")
    
    metrics = get_model_metrics(st.session_state.model, st.session_state.X_test, st.session_state.y_test)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Accuracy",
            f"{metrics['accuracy']:.2%}",
            "🎯",
            "#10b981"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "AUC Score",
            f"{metrics['auc_score']:.4f}",
            "📈",
            "#00d4ff"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card(
            "True Positives",
            f"{metrics['confusion_matrix'][1,1]}",
            "✅",
            "#10b981"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "False Positives",
            f"{metrics['confusion_matrix'][0,1]}",
            "❌",
            "#ef4444"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(plot_confusion_matrix(metrics['confusion_matrix']), use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Classification Report")
        report_text = metrics['classification_report']
        st.code(report_text, language='text')
    
    st.markdown("---")
    
    # Data Visualizations
    st.markdown("### 📊 Data Analysis Charts")
    
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "💰 Credit Analysis", "🏠 Housing & Risk"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_age_distribution(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_duration_analysis(df), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_credit_analysis(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_risk_comparison(df), use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_housing_risk(df), use_container_width=True)
        with col2:
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)

def show_dataset(df):
    """Dataset Page"""
    st.markdown("<h1>📈 Dataset Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(200,210,255,0.8); margin-bottom: 2rem;'>Browse and analyze the German Credit Dataset</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Statistics
    st.markdown("### 📊 Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Total Records",
            f"{len(df):,}",
            "📋",
            "#00d4ff"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card(
            "Features",
            f"{len(df.columns)}",
            "🔧",
            "#7c3aed"
        ), unsafe_allow_html=True)
    
    with col3:
        missing = df.isnull().sum().sum()
        st.markdown(create_metric_card(
            "Missing Values",
            f"{missing}",
            "⚠️",
            "#f59e0b"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card(
            "Memory Usage",
            f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            "💾",
            "#10b981"
        ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Preview
    st.markdown("### 📋 Dataset Preview")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        rows = st.slider("Number of rows to display:", 5, len(df), 10)
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    st.dataframe(df.head(rows), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Column Information
    st.markdown("### 🔍 Column Information")
    
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum()
    })
    
    st.dataframe(col_info, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Descriptive Statistics
    st.markdown("### 📈 Descriptive Statistics")
    
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.markdown("---")
    
    # Download Options
    st.markdown("### ⬇️ Download Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="credit_data.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name="credit_data.json",
            mime="application/json"
        )
    
    with col3:
        if st.button("🔄 Reload Dataset"):
            st.cache_data.clear()
            st.rerun()

def show_history():
    """Prediction History Page"""
    st.markdown("<h1>⏱️ Prediction History</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: rgba(200,210,255,0.8); margin-bottom: 2rem;'>View all predictions made in this session</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not st.session_state.prediction_history:
        st.info("No predictions yet. Go to the Prediction page to make your first prediction!")
        return
    
    # History Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Total Predictions",
            len(st.session_state.prediction_history),
            "📊",
            "#00d4ff"
        ), unsafe_allow_html=True)
    
    good_risk_count = sum(1 for h in st.session_state.prediction_history if h['prediction'] == 'Good Risk')
    with col2:
        st.markdown(create_metric_card(
            "Good Risk",
            good_risk_count,
            "✅",
            "#10b981"
        ), unsafe_allow_html=True)
    
    bad_risk_count = sum(1 for h in st.session_state.prediction_history if h['prediction'] == 'Bad Risk')
    with col3:
        st.markdown(create_metric_card(
            "Bad Risk",
            bad_risk_count,
            "⚠️",
            "#ef4444"
        ), unsafe_allow_html=True)
    
    if good_risk_count + bad_risk_count > 0:
        approval_rate = (good_risk_count / (good_risk_count + bad_risk_count)) * 100
        with col4:
            st.markdown(create_metric_card(
                "Approval Rate",
                f"{approval_rate:.1f}%",
                "��",
                "#f59e0b"
            ), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # History Table
    st.markdown("### 📋 Prediction Records")
    
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df['good_prob'] = (history_df['good_prob'] * 100).round(2).astype(str) + '%'
    history_df['bad_prob'] = (history_df['bad_prob'] * 100).round(2).astype(str) + '%'
    history_df['income'] = history_df['income'].apply(lambda x: f"₹{x:,.0f}")
    history_df['credit'] = history_df['credit'].apply(lambda x: f"₹{x:,.0f}")
    
    st.dataframe(history_df, use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.success("History cleared!")
            st.rerun()
    
    with col2:
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    
    with col3:
        st.info(f"📌 {len(st.session_state.prediction_history)} predictions recorded")

def show_about():
    """About Page"""
    st.markdown("<h1>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🎯 Project Overview</h3>
            <p>This is a <strong>premium AI-powered Life Insurance Risk Prediction System</strong> built using machine learning to assess customer financial risk and creditworthiness.</p>
            <p style="margin-top: 15px;"><strong>Purpose:</strong> Help financial institutions make data-driven lending decisions by predicting customer risk profiles.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🔬 Technology Stack</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>✅ <strong>Python 3.8+</strong></li>
                <li>✅ <strong>Streamlit</strong> - Web Framework</li>
                <li>✅ <strong>Scikit-learn</strong> - ML Library</li>
                <li>✅ <strong>Plotly</strong> - Data Visualization</li>
                <li>✅ <strong>Pandas & NumPy</strong> - Data Processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🤖 Machine Learning Model</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>📊 <strong>Algorithm:</strong> Logistic Regression</li>
                <li>📈 <strong>Accuracy:</strong> """ + f"{st.session_state.accuracy:.2%}" + """</li>
                <li>📋 <strong>Features:</strong> 5 (Age, Sex, Housing, Credit Amount, Duration)</li>
                <li>🎯 <strong>Target:</strong> Risk (Good/Bad)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>📊 Dataset Information</h3>
            <ul style="list-style-type: none; padding-left: 0;">
                <li>📁 <strong>Dataset:</strong> German Credit Data</li>
                <li>📝 <strong>Records:</strong> """ + f"{len(st.session_state.df):,}" + """</li>
                <li>🔧 <strong>Features:</strong> """ + f"{len(st.session_state.df.columns)}" + """</li>
                <li>✨ <strong>Quality:</strong> No missing values (cleaned)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="glass-card">
        <h3>📚 Feature Descriptions</h3>
        <ul style="list-style-type: none; padding-left: 0;">
            <li><strong>Age:</strong> Applicant's age in years (18-75)</li>
            <li><strong>Sex:</strong> Applicant's gender (Male/Female)</li>
            <li><strong>Housing:</strong> Current housing arrangement (Own/Rent/Free)</li>
            <li><strong>Credit Amount:</strong> Requested loan amount (₹)</li>
            <li><strong>Duration:</strong> Loan repayment period in months</li>
            <li><strong>Risk:</strong> Target variable (Good/Bad)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="glass-card">
        <h3>🎯 How It Works</h3>
        <ol style="padding-left: 20px;">
            <li><strong>Data Input:</strong> Enter applicant's financial information</li>
            <li><strong>Preprocessing:</strong> Data is encoded and normalized</li>
            <li><strong>Prediction:</strong> ML model processes the features</li>
            <li><strong>Analysis:</strong> Risk probability and business rules are evaluated</li>
            <li><strong>Recommendations:</strong> Smart suggestions are generated</li>
            <li><strong>Decision:</strong> Approve or Reject with confidence score</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>👨‍💻 Developer</h3>
            <p><strong>Name:</strong> Ronit Pattnaik</p>
            <p><strong>Role:</strong> AI/ML Developer</p>
            <p><strong>Focus:</strong> Predictive Analytics & Banking AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>📞 Contact Info</h3>
            <p><strong>Platform:</strong> GitHub</p>
            <p><strong>Repository:</strong> Life-insurance-prediction-project</p>
            <p><strong>Framework:</strong> Streamlit Cloud</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3>⚖️ Disclaimer</h3>
            <p>This system is for <strong>educational purposes</strong>. Real lending decisions should involve professional financial advisors and comply with regulatory requirements.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <h3>✨ Features</h3>
        <p>✅ Real-time Risk Prediction | 📊 Advanced Analytics | 📈 Historical Tracking | 💾 Data Export | 🎯 Smart Recommendations</p>
        <p style="margin-top: 15px; font-size: 0.9rem; color: rgba(200, 210, 255, 0.7);">Built with ❤️ using Streamlit | Powered by Machine Learning | Enterprise-Grade Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
