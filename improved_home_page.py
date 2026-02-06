"""
AutoEDA - Enhanced Automated Exploratory Data Analysis Platform
Production-ready internal tool for streamlining EDA across DS projects
Version 2.0 - Enterprise Grade
Improved Home Page UI with Simple and Elegant Design
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt
import uuid
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Import internal modules
import data_analysis_functions as analysis
import data_preprocessing_function as preprocessing

# Import enhanced modules
try:
    from enhanced_eda_functions import (
        display_data_quality_dashboard,
        display_correlation_analysis,
        display_distribution_comparison,
        display_outlier_detection_report,
        display_feature_importance_analysis,
        display_time_series_analysis,
        display_categorical_insights
    )
    ENHANCED_EDA_AVAILABLE = True
except ImportError:
    ENHANCED_EDA_AVAILABLE = False

try:
    from api_ui_component import render_postman_style_api_interface
    POSTMAN_UI_AVAILABLE = True
except ImportError:
    POSTMAN_UI_AVAILABLE = False

try:
    from credential_manager import CredentialManager, render_credential_manager_ui, get_loaded_credentials
    CREDENTIAL_MANAGER_AVAILABLE = True
except ImportError:
    CREDENTIAL_MANAGER_AVAILABLE = False

# Import all connectors
from autoeda.connectors.csv_connector import load_csv, load_excel
from autoeda.connectors.mysql_connector import load_mysql
from autoeda.connectors.postgres_connector import load_postgres
from autoeda.connectors.sqlite_connector import load_sqlite
from autoeda.connectors.mssql_connector import load_mssql
from autoeda.connectors.mongodb_connector import load_mongodb
from autoeda.connectors.s3_connector import load_s3, list_s3_objects
from autoeda.connectors.azure_blob_connector import load_azure_blob
from autoeda.connectors.gcs_connector import load_gcs

# Import enhanced REST API connector
try:
    from rest_api_connector import load_rest_api, test_api_connection
except ImportError:
    from autoeda.connectors.rest_api_connector import load_rest_api

# Page Configuration
st.set_page_config(
    page_icon="📊",
    page_title="AutoEDA - Internal DS Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with Simple and Elegant Design
st.markdown("""
    <style>
    /* Main Header Styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Welcome Section Styling */
    .welcome-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        letter-spacing: -0.5px;
    }
    
    .welcome-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .welcome-tagline {
        font-size: 1rem;
        opacity: 0.9;
        font-style: italic;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        padding-top: 1.5rem;
    }
    
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.8rem;
    }
    
    .feature-description {
        font-size: 0.95rem;
        color: #718096;
        line-height: 1.6;
    }
    
    .feature-list {
        list-style: none;
        padding-left: 0;
        margin-top: 1rem;
    }
    
    .feature-list li {
        padding: 0.5rem 0;
        color: #4a5568;
        font-size: 0.9rem;
    }
    
    .feature-list li:before {
        content: "✓ ";
        color: #667eea;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Info Box Styling */
    .info-banner {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin-bottom: 2rem;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.1);
    }
    
    .info-banner-text {
        color: #4c51bf;
        font-size: 1rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Custom Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f8f9fa;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        color: #6c757d;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #667eea;
        border-bottom: 3px solid #667eea;
        font-weight: 600;
    }
    
    /* Success/Warning/Info boxes */
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'new_df' not in st.session_state:
        st.session_state.new_df = None
    if 'source_dataset' not in st.session_state:
        st.session_state.source_dataset = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'api_response_metadata' not in st.session_state:
        st.session_state.api_response_metadata = {}
    if 'eda_findings' not in st.session_state:
        st.session_state.eda_findings = []

init_session_state()

# [Include all the analysis functions from the original code]
# perform_univariate_analysis, perform_bivariate_analysis, etc.
# [Copy all functions from the original code here]

# ============================================================================
# SIDEBAR - DATA SOURCE CONNECTOR (Keep as is from original code)
# ============================================================================

def render_sidebar():
    """Render sidebar with data source selection and connection options"""
    
    st.sidebar.markdown("# 📊 AutoEDA")
    st.sidebar.markdown("*Internal DS Platform*")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("🔌 Data Source Connector")
    
    # Data source categories
    file_sources = ["CSV/Excel File"]
    database_sources = ["MySQL", "PostgreSQL", "SQLite", "MSSQL", "MongoDB"]
    cloud_sources = ["AWS S3", "Azure Blob Storage", "Google Cloud Storage"]
    api_sources = ["REST API"]
    
    source_category = st.sidebar.radio(
        "Select Source Type",
        ["📁 Files", "💾 Databases", "☁️ Cloud Storage", "🌐 APIs"],
        key="sidebar_source_category_radio"
    )
    
    if source_category == "📁 Files":
        data_source = st.sidebar.selectbox("Choose File Type", file_sources, key="file_type_select")
    elif source_category == "💾 Databases":
        data_source = st.sidebar.selectbox("Choose Database", database_sources, key="database_select")
    elif source_category == "☁️ Cloud Storage":
        data_source = st.sidebar.selectbox("Choose Cloud Provider", cloud_sources, key="cloud_select")
    else:
        data_source = st.sidebar.selectbox("Choose API Type", api_sources, key="api_select")
    
    st.sidebar.markdown("---")
    
    return data_source

# [Include all other sidebar functions - handle_file_upload, handle_mysql_connection, etc.]

# ============================================================================
# ENHANCED WELCOME SCREEN
# ============================================================================

def render_welcome_screen():
    """Render elegant welcome screen"""
    
    # Main Header
    st.markdown('<p class="main-header">AutoEDA Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Internal Data Science Platform for Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # Welcome Banner
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">Welcome to AutoEDA</div>
            <div class="welcome-subtitle">Your comprehensive platform for automated exploratory data analysis and data preprocessing</div>
            <div class="welcome-tagline">"If we do something for the first time, we do it for the first time only once..."</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Info Banner
    st.markdown("""
        <div class="info-banner">
            <p class="info-banner-text">
                <span>👈</span>
                <span>Please connect to a data source and load data from the sidebar to get started</span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown('<h2 class="section-header">🚀 Key Features</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">📥</span>
                <div class="feature-title">Multiple Data Sources</div>
                <div class="feature-description">Connect and load data from various sources seamlessly</div>
                <ul class="feature-list">
                    <li>CSV & Excel Files</li>
                    <li>SQL Databases (MySQL, PostgreSQL, SQLite, MSSQL)</li>
                    <li>NoSQL (MongoDB)</li>
                    <li>Cloud Storage (AWS S3, Azure Blob, Google Cloud Storage)</li>
                    <li>REST APIs with custom configurations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">🔍</span>
                <div class="feature-title">Advanced EDA Capabilities</div>
                <div class="feature-description">Comprehensive exploratory data analysis tools</div>
                <ul class="feature-list">
                    <li>Univariate Analysis - Distribution & Statistics</li>
                    <li>Bivariate Analysis - Correlation & Relationships</li>
                    <li>Multivariate Analysis - PCA, Pair Plots, 3D Visualizations</li>
                    <li>Outlier Detection - IQR, Z-Score, Isolation Forest</li>
                    <li>Data Quality Dashboard</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">🛠️</span>
                <div class="feature-title">Data Preprocessing</div>
                <div class="feature-description">Clean and transform your data efficiently</div>
                <ul class="feature-list">
                    <li>Handle Missing Data (Remove, Fill, Impute)</li>
                    <li>Encode Categorical Variables</li>
                    <li>Feature Scaling (Standardization, Min-Max, Robust)</li>
                    <li>Outlier Treatment</li>
                    <li>Feature Engineering</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">⚡</span>
                <div class="feature-title">Advanced Features</div>
                <div class="feature-description">Powerful tools for data professionals</div>
                <ul class="feature-list">
                    <li>Dataset Joining & Merging</li>
                    <li>Custom SQL Query Editor</li>
                    <li>Export Preprocessed Data (CSV, Excel)</li>
                    <li>Version Control for Datasets</li>
                    <li>Comprehensive Summary Reports</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Getting Started Section
    st.markdown('<h2 class="section-header">📖 Getting Started</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">1️⃣</span>
                <div class="feature-title">Connect</div>
                <div class="feature-description">Select your data source from the sidebar</div>
            </div>
        """, unsafe_allow_html=True)
    
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">3️⃣</span>
                <div class="feature-title">Load</div>
                <div class="feature-description">Import your dataset</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="feature-card">
                <span class="feature-icon">4️⃣</span>
                <div class="feature-title">Analyze</div>
                <div class="feature-description">Explore and preprocess your data</div>
            </div>
        """, unsafe_allow_html=True)
    

# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================

def main():
    """Main application entry point"""
    
    # Render sidebar and get data source
    data_source = render_sidebar()
    
    # Handle different data sources based on selection
    # [Include all data source handling code from original]
    
    # Main content area
    if st.session_state.datasets:
        # Dataset loaded - show analysis interface
        # [Include all analysis code from original]
        pass
    else:
        # No dataset loaded - show welcome screen
        render_welcome_screen()

if __name__ == "__main__":
    main()