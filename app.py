import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import hashlib
import os

# --- User Authentication ---
USERS_FILE = 'users.csv'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    df = pd.read_csv(USERS_FILE)
    return dict(zip(df['username'], df['password']))

def save_user(username, password_hash):
    if os.path.exists(USERS_FILE):
        df = pd.read_csv(USERS_FILE)
        df = pd.concat([df, pd.DataFrame({'username': [username], 'password': [password_hash]})], ignore_index=True)
    else:
        df = pd.DataFrame({'username': [username], 'password': [password_hash]})
    df.to_csv(USERS_FILE, index=False)

# --- Enhanced Login/Registration Page ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'show_registration' not in st.session_state:
    st.session_state.show_registration = False

if not st.session_state.logged_in:
    st.markdown("""
    <style>
    .login-card {
        background-color: white;
        padding: 2.5rem 2rem 2rem 2rem;
        border-radius: 18px;
        box-shadow: 0 4px 24px 0 rgba(0,0,0,0.08);
        max-width: 400px;
        margin: 40px auto 0 auto;
    }
    .login-title {
        color: #0072C6;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .login-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    </style>
    <div class="login-card">
        <div class="login-logo">
            <img src="https://img.icons8.com/color/96/000000/5g.png" width="64"/>
        </div>
        <div class="login-title">5G Energy Optimizer</div>
    </div>
    """, unsafe_allow_html=True)
    login_container = st.container()
    with login_container:
        if st.session_state.show_registration:
            with st.form(key='register_form'):
                st.subheader('📝 Register')
                reg_username = st.text_input('Choose a username', key='reg_username')
                reg_password = st.text_input('Choose a password', type='password', key='reg_password')
                reg_password2 = st.text_input('Confirm password', type='password', key='reg_password2')
                register_btn = st.form_submit_button('Register')
                if register_btn:
                    users = load_users()
                    if reg_username in users:
                        st.error('Username already exists!')
                    elif reg_password != reg_password2:
                        st.error('Passwords do not match!')
                    elif not reg_username or not reg_password:
                        st.error('Please fill all fields!')
                    else:
                        save_user(reg_username, hash_password(reg_password))
                        st.success('Registration successful! Please log in.')
                        st.session_state.show_registration = False
                        st.stop()
            if st.button('Back to Login'):
                st.session_state.show_registration = False
                st.stop()
        else:
            with st.form(key='login_form'):
                st.subheader('🔐 Login')
                login_username = st.text_input('Username', key='login_username')
                login_password = st.text_input('Password', type='password', key='login_password')
                login_btn = st.form_submit_button('Login')
                if login_btn:
                    users = load_users()
                    if login_username in users and users[login_username] == hash_password(login_password):
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.success(f'Welcome, {login_username}!')
                        st.stop()
                    else:
                        st.error('Invalid username or password!')
            if st.button('Register New Account'):
                st.session_state.show_registration = True
                st.stop()
    st.stop()

# --- Streamlit App Layout ---
st.set_page_config(page_title="ML-based Energy Efficiency Optimization for 5G Networks", layout="wide", page_icon="🔋")

# --- Top Right User Menu ---
if st.session_state.get('logged_in', False):
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button(f"👤 {st.session_state.get('username', '')}"):
            st.session_state.show_user_menu = not st.session_state.get('show_user_menu', False)
        if st.session_state.get('show_user_menu', False):
            option = st.radio(
                "Account Options",
                ("View Profile", "Logout"),
                key="user_menu_radio"
            )
            if option == "Logout":
                st.session_state.logged_in = False
                st.session_state.username = ''
                st.session_state.show_user_menu = False
                st.stop()
            elif option == "View Profile":
                st.session_state.show_profile = True
    if st.session_state.get('show_profile', False):
        st.markdown("---")
        st.markdown(f"### 👤 Profile for <span style='color:#0072C6'>{st.session_state.get('username','')}</span>", unsafe_allow_html=True)
        st.write("This is a placeholder for user profile details.")
        if st.button("Close Profile"):
            st.session_state.show_profile = False

# --- Sidebar Branding ---
st.sidebar.image("https://img.icons8.com/color/96/000000/5g.png", width=80)
st.sidebar.markdown("## <span style='color:#0072C6'>5G Energy Optimizer</span>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
pages = [
    "Data Upload & ML",
    "Networks and Beyond",
    "Objectives"
]
page = st.sidebar.radio("🌐 Navigation", pages, index=0)

st.markdown("""
<style>
    .main {background-color: #f7fafd;}
    .stButton>button {background-color: #0072C6; color: white; border-radius: 8px;}
    .st-bb {background-color: #e3f2fd;}
    .st-cq {background-color: #e3f2fd;}
</style>
""", unsafe_allow_html=True)

st.title("🔋 ML-based Energy Efficiency Optimization for 5G Networks and Beyond")

if page == "Data Upload & ML":
    st.header("📊 Upload Your Energy Dataset")
    st.markdown('<div style="background-color: white; padding: 10px; border-radius: 8px;"><span style="color: black; font-size: 16px;">Upload a CSV file containing your energy dataset. Select features and target for ML-based energy consumption prediction.</span></div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📁 Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.write("### 👀 Preview of uploaded data:")
            st.dataframe(user_data.head(), use_container_width=True)
            
            numeric_columns = user_data.select_dtypes(include=[np.number]).columns.tolist()
            all_columns = user_data.columns.tolist()
            st.markdown("---")
            with st.expander("🔧 Select Features and Target", expanded=True):
                feature_cols = st.multiselect("Select feature columns (numeric only)", numeric_columns, default=numeric_columns[:-1] if len(numeric_columns) > 1 else numeric_columns)
                target_col = st.selectbox("Select target column", all_columns, index=len(all_columns)-1)
            if feature_cols and target_col:
                if not np.issubdtype(user_data[target_col].dtype, np.number):
                    st.warning("⚠️ Selected target column is not numeric. Please select a numeric target column for regression.")
                else:
                    X = user_data[feature_cols]
                    y = user_data[target_col]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    st.markdown("---")
                    with st.expander("📈 Show Model Performance", expanded=True):
                        st.subheader("Model Performance")
                        st.success(f"Mean Squared Error: {mse:.2f}")
                        fig, ax = plt.subplots(figsize=(7, 5))
                        ax.scatter(y_test, y_pred, alpha=0.8, c='#0072C6', edgecolor='k', label='Predictions')
                        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
                        ax.set_xlabel('Actual', fontsize=12)
                        ax.set_ylabel('Predicted', fontsize=12)
                        ax.set_title('Actual vs Predicted', fontsize=14, color='#0072C6')
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.5)
                        st.pyplot(fig)
                        st.info("Blue dots: Model predictions. Red dashed: Ideal fit.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.markdown('<div style="background-color: white; padding: 10px; border-radius: 8px;"><span style="color: black; font-size: 16px;">Please upload a CSV file to proceed.</span></div>', unsafe_allow_html=True)

elif page == "Networks and Beyond":
    st.header("🌍 Networks and Beyond")
    st.subheader("🤖 Real-time ML Inference")
    st.write("Interact with a large language model for energy efficiency queries or brainstorming.")
    st.markdown("---")

    # Make the text input color black
    st.markdown("""
    <style>
    input[type="text"] {
        color: black !important;
    }
    </style>
    """, unsafe_allow_html=True)

    user_input = st.text_input("💬 Ask a question about energy efficiency in 5G networks:")

    if user_input:
        client = OpenAI(
            api_key = "504d7cdd-c8ac-4ab6-bada-90e9a77d4adb",
            base_url = "https://api.kluster.ai/v1"
        )
        with st.spinner("Getting response from the model..."):
            try:
                completion = client.chat.completions.create(
                    model = "klusterai/Meta-Llama-3.1-8B-Instruct-Turbo",
                    messages = [
                        { "role": "user", "content": user_input }
                    ]
                )
                st.success(completion.choices[0].message.content)
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown("---")
    st.header("🚀 Future Research Directions")
    st.markdown("""
- 🛡️ Integration of federated learning for privacy-preserving energy optimization.
- 🛰️ Use of digital twins for real-time network simulation and optimization.
- 🧩 Cross-layer ML approaches for holistic energy management.
- 🤖 Application of generative AI for predictive maintenance and anomaly detection.
- ⚛️ Exploration of quantum ML for ultra-efficient network operations.
""")

elif page == "Objectives":
    st.header("🎯 Objectives")
    st.markdown("""
1. **Identify the challenges in energy consumption in 5G and beyond networks**: 
   - Sources: Base stations, user equipment, backhaul, etc.
   - Limitations: Static power allocation, lack of real-time adaptation, etc.
2. **Investigate various ML algorithms**: 
   - Supervised learning, reinforcement learning, deep learning for energy efficiency.
3. **Propose a comprehensive ML framework**: 
   - Dynamic power management, resource allocation, energy-efficient operation.
4. **Evaluate ML techniques**: 
   - Experiments/simulations to assess impact on energy, performance, scalability.
5. **Propose future research directions**: 
   - Recommendations for advancing ML-based energy optimization in future networks.
""")
    st.markdown("---")
    st.info("Explore each section using the navigation sidebar for a complete experience.")