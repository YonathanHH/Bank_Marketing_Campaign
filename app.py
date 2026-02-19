import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

st.set_page_config(page_title="Bank Campaign Marketing Predictor", layout="wide")
def set_bg_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        [data-testid="stSidebar"] {{
            background-color: #086908
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg_color("#540505")
MODEL_FILE = '01_bank_marketing_model.sav'

# Feature definitions based on provided docs
CATEGORICAL_FEATURES = [
    'job', 'marital', 'education', 'housing', 'loan', 
    'is_default_status_known', 'contact', 'month', 
    'day_of_week', 'was_contacted_before', 'poutcome'
]
NUMERICAL_FEATURES = [
    'age', 'campaign', 'previous', 
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]

JOB_OPTS = ['admin', 'blue-collar', 'technician', 'services', 'management', 
            'retired', 'entrepreneur', 'self-employed', 'housemaid', 
            'unemployed', 'student']
MARITAL_OPTS = ['married', 'single', 'divorced']
EDU_OPTS = ['university degree', 'high school', 'basic 9 years', 'professional course', 
            'basic 4 years', 'basic 6 years', 'illiterate']
YES_NO_OPTS = ['yes', 'no']
CONTACT_OPTS = ['telephone', 'cellular']
MONTH_OPTS = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
DAY_OPTS = ['mon', 'tue', 'wed', 'thu', 'fri']
POUTCOME_OPTS = ['nonexistent', 'failure', 'success']


def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        return None

def build_pipeline():

    
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=3)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputerWrapper(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # *Adjustment*: Since KNNImputer requires numeric input, applying it to raw categorical strings 
    # directly causes errors. For the strict requirements, we usually encode first. 
    # To keep this robust for the app, I'm using a standard preprocessing flow that matches the intent.

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ])

    # LGBM Parameters 
    clf = lgb.LGBMClassifier(
        is_unbalance=False,
        learning_rate=0.05,
        n_estimators=300,
        num_leaves=30,
        random_state=42
    )

    return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class SimpleImputerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='most_frequent'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
    def fit(self, X, y=None):
        self.imputer.fit(X, y)
        return self
    def transform(self, X):
        return self.imputer.transform(X)

def main():

    # Sidebar Navigation
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    st.sidebar.title("Navigation")
    if st.sidebar.button("Home"):
        st.session_state.page = "Home"
    if st.sidebar.button("  Single Prediction"):
        st.session_state.page = "Single Prediction"
    if st.sidebar.button("Batch Prediction"):
        st.session_state.page = "Batch Prediction (CSV)"
    if st.sidebar.button("Train Model"):
        st.session_state.page = "Train Model"
    if st.sidebar.button("Model Documentation"):
        st.session_state.page = "Model Specs"
    if st.sidebar.button("Data Dictionary"):
        st.session_state.page = "Data Dictionary"
    if st.sidebar.button("Macroeconomic Analysis"):
        st.session_state.page = "Macroeconomic Analysis"

    choice = st.session_state.page

    # Load Model
    model = load_model()

    if choice == "Home":
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0;'>
                <h1 style='font-size: 3.5rem; color: #FFD700; margin-bottom: 0.5rem;'>
                    🏦 Bank Campaign Marketing Predictor
                </h1>
                <p style='font-size: 1.3rem; color: #FFFFFF; margin-top: 0;'>
                    Predict Term Deposit Subscription with Machine Learning
                </p>
            </div>
        """, unsafe_allow_html=True)
    
        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ###  Welcome to the Bank Marketing Campaign Predictor!
        
            This application uses **Machine Learning (LightGBM)** to predict whether a customer 
            will subscribe to a term deposit based on their demographic information, 
            previous campaign interactions, and macroeconomic indicators.
        
            ####  What Can You Do Here?
        
            - **Single Prediction:** Predict subscription likelihood for individual customers
            - **Batch Prediction:** Upload CSV files for bulk predictions
            - **Train Model:** Train or retrain the model with your own data
            - **Model Documentation:** View model performance metrics and specifications
            - **Data Dictionary:** Understand all features used in the model
            - **Macroeconomic Analysis:** Learn about economic context during data collection
        
            #### Model Performance
            - **Accuracy:** 84.28% | **F1-Score:** 48.53% | **ROC-AUC:** 81.52%
            """)
    
        with col2:
            st.info("""
            **Quick Facts**
        
            **Model:** LightGBM Classifier
        
            **Dataset:** 41,188 records
        
            **Features:** 20 input variables
        
            **Created by:**
            - Fatimah Azzahra
            - Tengku Arika Hazera
            - Yonathan Hary Hutagalung
        
            *Purwadhika Data Science Final Project*
            """)
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 10px; height: 250px;'>
                <h3 style='color: #FFD700;'> Make Predictions</h3>
                <p style='color: #FFFFFF;'>
                    Predict subscription probability for individual customers or upload CSV files for batch predictions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
            if st.button(" Single Prediction", use_container_width=True, type="primary"):
                st.session_state.page = "Single Prediction"
                st.rerun()
        
            if st.button(" Batch Prediction", use_container_width=True):
                st.session_state.page = "Batch Prediction (CSV)"
                st.rerun()
        with col2:
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 10px; height: 250px;'>
                <h3 style='color: #FFD700;'> Model Management</h3>
                <p style='color: #FFFFFF;'>
                    Train or retrain the model with custom data, and view comprehensive model documentation.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
            if st.button(" Train Model", use_container_width=True, type="primary"):
                st.session_state.page = "Train Model"
                st.rerun()
        
            if st.button(" Model Documentation", use_container_width=True):
                st.session_state.page = "Model Specs"
                st.rerun()
        
        with col3:
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 10px; height: 250px;'>
                <h3 style='color: #FFD700;'> Learn More</h3>
                <p style='color: #FFFFFF;'>
                    Explore the data dictionary and understand macroeconomic factors affecting predictions.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
            if st.button(" Data Dictionary", use_container_width=True, type="primary"):
                st.session_state.page = "Data Dictionary"
                st.rerun()
        
            if st.button(" Macroeconomic Analysis", use_container_width=True):
                st.session_state.page = "Macroeconomic Analysis"
                st.rerun()

    if choice == "Single Prediction":
        st.header("Single Customer Prediction")
        
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()
        if model is None:
            st.warning(f"⚠️ Model file `{MODEL_FILE}` not found. Please go to 'Train Model' first.")
        else:
            col1, col2, col3 = st.columns(3)

            # Personal Info
            with col1:
                st.subheader("Personal Info")
                age = st.number_input("Age", min_value=17, max_value=98, value=30, help="Range: 17-98 ")
                job = st.selectbox("Job", JOB_OPTS, help="Type of job")
                marital = st.selectbox("Marital Status", MARITAL_OPTS)
                education = st.selectbox("Education", EDU_OPTS)
                housing = st.selectbox("Has Housing Loan?", YES_NO_OPTS)
                loan = st.selectbox("Has Personal Loan?", YES_NO_OPTS)
                is_default = st.selectbox("Is Default Status Known?", YES_NO_OPTS)

            # Last Campaign Info 
            with col2:
                st.subheader("Last Campaign")
                contact = st.selectbox("Contact Type", CONTACT_OPTS)
                month = st.selectbox("Month", MONTH_OPTS)
                day = st.selectbox("Day of Week", DAY_OPTS)
                was_contacted = st.selectbox("Was Contacted Before?", YES_NO_OPTS)
                campaign = st.number_input("Campaign Contacts", min_value=1, max_value=15, value=1, help="Range: 1-15 ")
                previous = st.number_input("Previous Contacts", min_value=0, max_value=7, value=0, help="Range: 0-7 ")
                poutcome = st.selectbox("Previous Outcome", POUTCOME_OPTS)

            # Macroeconomic Indicators 
            with col3:
                st.subheader("Macro Indicators")
                cons_idx = st.number_input("Cons. Conf. Index", min_value=-50.8, max_value=-26.9, value=-40.0, step=0.1, help="Range: -50.8 to -26.9 ")
                euribor = st.number_input("Euribor 3m", min_value=0.634, max_value=5.045, value=1.0, step=0.001, help="Range: 0.634 to 5.045")
                nr_employed = st.number_input("Nr. Employed", min_value=4963.6, max_value=5228.1, value=5000.0, step=0.1, help="Range: 4963.6 to 5228.1")

            # Create Input DataFrame
            input_data = pd.DataFrame({
                'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
                'housing': [housing], 'loan': [loan], 'is_default_status_known': [is_default],
                'contact': [contact], 'month': [month], 'day_of_week': [day],
                'was_contacted_before': [was_contacted],
                'campaign': [campaign], 'previous': [previous], 'poutcome': [poutcome],
                'cons.conf.idx': [cons_idx], 'euribor3m': [euribor], 'nr.employed': [nr_employed]
            })

            if st.button("Predict Subscription"):
                prediction = model.predict(input_data)[0]
                proba = model.predict_proba(input_data)[0][1]
                
                if proba >= 0.5:
                    st.success(f"Prediction: **Yes, the customer likely to subscribe* - Probability: {proba:.2%}")
                else:
                    st.error(f"Prediction: **No, the customer are not likely to subscribe** - Probability: {proba:.2%}")
                
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=proba * 100,           # convert to percentage
                        number={"suffix": "%"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#22c55e" if proba >= 0.5 else "#ef4444"},
                            "steps": [
                                {"range": [0, 50], "color": "#7f1d1d"},
                                {"range": [50, 75], "color": "#92400e"},
                                {"range": [75, 100], "color": "#14532d"},
                            ],
                            "threshold": {
                                "line": {"color": "white", "width": 4},
                                "thickness": 0.8,
                                "value": proba * 100,
                            },
                        },
                        title={"text": "Subscription Probability"}
                    )
                )

                st.plotly_chart(fig, use_container_width=True)


    elif choice == "Batch Prediction (CSV)":
        st.header("Upload CSV for Batch Prediction")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()        
        
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", data.head())
            
            if model is None:
                st.error("Model not loaded. Please train the model first.")
            else:
                if st.button("Predict All"):
                    try:
                        # Ensure columns match
                        predictions = model.predict(data)
                        data['Prediction'] = predictions
                        
                        st.success("Prediction complete!")
                        st.dataframe(data)
                        
                        # Download Button
                        csv = data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name='bank_predictions.csv',
                            mime='text/csv',
                        )
                    except Exception as e:
                        st.error(f"Error during prediction. Please ensure CSV columns match the model requirements. Details: {e}")


    elif choice == "Train Model":
        trained_model = 'bank_model_trained.sav'
        st.header("Train/Retrain Model")
        st.info(f"This will train a new LGBM model with specified parameters and save it as `{trained_model}`.")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()

        train_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
        
        if train_file:
            df = pd.read_csv(train_file)
            st.write(f"Training data shape: {df.shape}")
            
            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    # Separate features and target
                    X = df.drop(columns=['y'])
                    y = df['y']
                    
                    # Build and Fit Pipeline
                    pipeline = build_pipeline()
                    pipeline.fit(X, y)
                    
                    # Save Model
                    joblib.dump(pipeline, trained_model)
                    st.success(f"Model trained and saved to `{trained_model}`.")
                    y_pred = pipeline.predict(X)
                    acc = accuracy_score(y, y_pred)
                    prc = precision_score(y, y_pred)
                    rec = recall_score(y, y_pred)
                    f1 = f1_score(y, y_pred)
                    f2 = (5 * prc * rec) / (4 * prc + rec)
                    auc_roc = roc_auc_score(y, y_pred)                    
                    st.metric("Training Accuracy", f"{acc:.4f}")
                    st.metric("Training Precision", f"{prc:.4f}")
                    st.metric("Training Recall", f"{rec:.4f}")
                    st.metric("Training F1-Score", f"{f1:.4f}")
                    st.metric("Training F2-Score", f"{f2:.4f}")
                    st.metric("Training ROC-AUC", f"{auc_roc:.4f}")



    elif choice == "Model Specs":
        st.header("Model Documentations")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()

        st.markdown("Created by Fatimah Azzahra, Tengku Arika Hazera, Yonathan Hary Hutagalung part of Purwadhika Data Science Final project.")
        st.markdown("---")
        st.markdown("Information based on Model Documentation:")
        
        col_1, col_2, col_3 = st.columns(3)
        
        with col_1:
            st.subheader("Performance Metrics")
            st.write(f"**Accuracy:** 0.8428 ")
            st.write(f"**Precision:** 0.3848 ")
            st.write(f"**Recall:** 0.6471 ")
            st.write(f"**F1-Score:** 0.4853 ")
            st.write(f"**F2-Score:** 0.5756 ")
            st.write(f"**ROC-AUC:** 0.8152 ")
            st.write(f"**PR-AUC:** 0.4815")

        with col_2:
            st.subheader("Configuration")
            st.write("**Model:** LGBM Classifier ")
            st.code("""
            classifier__is_unbalance: False
            classifier__learning_rate: 0.03
            classifier__n_estimators: 300
            classifier__max_depth: 5
            """, language="python")
            
            st.subheader("Pipeline")
            st.markdown("""
            * **Numerical:** Standard Scaler, KNN Imputer
            * **Categorical:** OneHotEncoder, KNN Imputer
            """)
        with col_3:
            st.subheader("Dataset Information:")
            st.write("**Dataset:** Bank Campaign Marketing Dataset")
            st.write("**Source:** https://archive.ics.uci.edu/ml/datasets/bank+marketing")
            st.write("**Size:** 41,188.0 rows")
            st.write("**Features:** 20 input variables + 1 target (y)")
            st.write()
    
    elif choice =="Data Dictionary":
        st.header("Data Dictionary")
        st.write("Reference guide for all features used in the bank marketing campaign model.")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()        

        with st.expander("1. Bank Client Data"):
            st.markdown("""
            * **Age:** Numerical age of the client.
            * **Job:** Type of job (e.g., admin, blue-collar, entrepreneur, etc.).
            * **Marital:** Marital status (Note: 'divorced' also includes widowed).
            * **Education:** Level of education (e.g., university degree, high school, illiterate).
            * **Is default status known:** Whether we have the information about customers' default status .
            * **Has a housing loan?:** Whether Customer have a housing loan.
            * **Has a personal loan?:** Whether Customer have a personal loan.
            """)
        with st.expander("2. Last Contact Information"):
            st.markdown("""
            * **Contact:** Contact communication type (cellular, telephone).
            * **Month:** Last contact month of the year.
            * **Day_of_week:** Last contact day of the week.
            """)
        with st.expander("3. Campaign & Social-Economic Context"):
            st.markdown("""
            * **Campaign:** Number of contacts performed during this campaign for this client.
            * **Pdays:** Number of days since client was last contacted (999 means not previously contacted).
            * **Previous:** Number of contacts performed before this campaign.
            * **Poutcome:** Outcome of the previous marketing campaign.
            * **Cons.Conf.Index**: Customer confidence index (numeric).
            * **Euribor3m:** Euribor 3 month rate - daily indicator.
            * **Nr.Employed:** Number of employees - quarterly indicator.
            """)

    elif choice == "Macroeconomic Analysis":
        st.header("Macroeconomic Context & Indicators")
        if st.button("⬅️ Back to Home"):
            st.session_state.page = "Home"
            st.rerun()
        st.image("https://www.historyhit.com/app/uploads/2021/10/crash.jpg", width=300)
        st.info("The model utilizes data from the 2008 economic crisis period.")
    
        st.markdown("""
        The dataset was collected during the 2008 financial crisis, which was characterized by high interest rates, low customer confidence, and a growing unemployment rate. 
        While we initially hypothesized that higher interest rates would encourage term deposits due to higher gains, the data suggests the opposite reaction occurred.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Monetary Drivers")
            st.write("**Euribor 3m:** Measures interest rates. Higher rates indicate more potential profit for the customer, serving as the 'Trigger' for bank cash desperation.")
            st.write("**Cons. Price Index (CPI):** Measures inflation (though less impactful without specific timeframes).")

            st.subheader("2. Economic Health")
            st.write("**Nr. Employed:** Measures job market conditions. This continuous variable is highly favorable for Machine Learning accuracy.")
    
        with col2:
            st.subheader("3. Psychological Factors")
            st.write("**Cons. Conf. Index:** Measures how confident and optimistic customers are about spending or saving.")

            st.divider()
            st.subheader("The Crisis Timeline (2008-2010)")
            st.warning("""
            **The Chain Reaction:**
            1. **The Trigger:** Banks became desperate for cash, causing **EURIBOR** to spike.
            2. **The Reaction:** Crisis hit consumer confidence (**Cons.Conf.Idx**), lowering the encouragement to save.
            3. **The Fallout:** The financial crisis eventually hit the job market (**Nr.Employed**) between 2009 and 2010.
            """)
    
if __name__ == '__main__':
    main()