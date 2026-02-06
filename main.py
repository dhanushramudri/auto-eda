"""
AutoEDA - Enhanced Automated Exploratory Data Analysis Platform
Production-ready internal tool for streamlining EDA across DS projects
Version 2.0 - Enterprise Grade
Enhanced with comprehensive EDA techniques and best practices
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
    st.warning("Enhanced EDA functions not available. Using basic features only.")

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

# Custom CSS for better UI
st.markdown("""
    <style>
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .eda-insights {
        padding: 1rem;
        background-color: #e7f3ff;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        border-bottom: 2px solid #1f77b4;
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

# ============================================================================
# EDA ANALYSIS FUNCTIONS - NEW FEATURES
# ============================================================================

def perform_univariate_analysis(df):
    """Perform comprehensive univariate analysis on dataset"""
    st.markdown("## 📊 Univariate Analysis")
    st.markdown("*Analyzing individual variables to understand their distributions and characteristics*")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if num_cols:
        st.subheader("Numerical Variables Distribution")
        selected_num = st.multiselect("Select numerical columns to analyze", num_cols, default=num_cols[:3])
        
        if selected_num:
            cols_per_row = 2
            cols = st.columns(cols_per_row)
            
            for idx, col in enumerate(selected_num):
                with cols[idx % cols_per_row]:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram with KDE
                    axes[0].hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                    axes[0].set_title(f'Histogram: {col}', fontweight='bold')
                    axes[0].set_xlabel('Value')
                    axes[0].set_ylabel('Frequency')
                    
                    # Box plot
                    axes[1].boxplot(df[col].dropna())
                    axes[1].set_title(f'Box Plot: {col}', fontweight='bold')
                    axes[1].set_ylabel('Value')
                    axes[1].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Statistics
                    col_stats = df[col].describe()
                    col_stats['skewness'] = df[col].skew()
                    col_stats['kurtosis'] = df[col].kurtosis()
                    
                    st.markdown(f"**Statistics for {col}:**")
                    st.dataframe(col_stats.to_frame(), use_container_width=True)
    
    if cat_cols:
        st.subheader("Categorical Variables Distribution")
        selected_cat = st.multiselect("Select categorical columns to analyze", cat_cols, default=cat_cols[:2])
        
        if selected_cat:
            for col in selected_cat:
                fig, ax = plt.subplots(figsize=(10, 4))
                value_counts = df[col].value_counts()
                value_counts.plot(kind='bar', ax=ax, color='teal', edgecolor='black', alpha=0.7)
                ax.set_title(f'Value Counts: {col}', fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.grid(alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistics
                st.markdown(f"**Statistics for {col}:**")
                cat_stats = pd.DataFrame({
                    'Unique Values': [df[col].nunique()],
                    'Missing Values': [df[col].isnull().sum()],
                    'Most Common': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'],
                    'Frequency': [df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0]
                })
                st.dataframe(cat_stats, use_container_width=True)

def perform_bivariate_analysis(df):
    """Perform comprehensive bivariate analysis"""
    st.markdown("## 🔗 Bivariate Analysis")
    st.markdown("*Analyzing relationships between pairs of variables*")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    analysis_type = st.radio(
        "Select Analysis Type",
        ["Numerical vs Numerical", "Categorical vs Categorical", "Numerical vs Categorical"],
        horizontal=True
    )
    
    if analysis_type == "Numerical vs Numerical" and len(num_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Select first variable", num_cols, key="num_var1")
        with col2:
            var2 = st.selectbox("Select second variable", num_cols, key="num_var2", index=1 if len(num_cols) > 1 else 0)
        
        if var1 != var2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            axes[0].scatter(df[var1], df[var2], alpha=0.6, color='steelblue', edgecolor='black')
            axes[0].set_xlabel(var1, fontweight='bold')
            axes[0].set_ylabel(var2, fontweight='bold')
            axes[0].set_title(f'Scatter Plot: {var1} vs {var2}', fontweight='bold')
            axes[0].grid(alpha=0.3)
            
            # Hexbin plot for dense data
            hb = axes[1].hexbin(df[var1], df[var2], gridsize=15, cmap='Blues')
            axes[1].set_xlabel(var1, fontweight='bold')
            axes[1].set_ylabel(var2, fontweight='bold')
            axes[1].set_title(f'Density Plot: {var1} vs {var2}', fontweight='bold')
            plt.colorbar(hb, ax=axes[1], label='Count')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Correlation
            corr = df[var1].corr(df[var2])
            st.markdown(f"**Pearson Correlation Coefficient:** `{corr:.4f}`")
    
    elif analysis_type == "Categorical vs Categorical" and len(cat_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            cat_var1 = st.selectbox("Select first categorical variable", cat_cols, key="cat_var1")
        with col2:
            cat_var2 = st.selectbox("Select second categorical variable", cat_cols, key="cat_var2", index=1 if len(cat_cols) > 1 else 0)
        
        if cat_var1 != cat_var2:
            # Cross tabulation
            crosstab = pd.crosstab(df[cat_var1], df[cat_var2])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            crosstab.plot(kind='bar', ax=ax, edgecolor='black')
            ax.set_title(f'Cross Tabulation: {cat_var1} vs {cat_var2}', fontweight='bold')
            ax.set_xlabel(cat_var1)
            ax.set_ylabel('Count')
            ax.legend(title=cat_var2, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("**Cross Tabulation Table:**")
            st.dataframe(crosstab, use_container_width=True)
    
    elif analysis_type == "Numerical vs Categorical" and num_cols and cat_cols:
        col1, col2 = st.columns(2)
        with col1:
            num_var = st.selectbox("Select numerical variable", num_cols, key="num_cat_var")
        with col2:
            cat_var = st.selectbox("Select categorical variable", cat_cols, key="cat_num_var")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        df.boxplot(column=num_var, by=cat_var, ax=axes[0])
        axes[0].set_title(f'Box Plot: {num_var} by {cat_var}', fontweight='bold')
        axes[0].set_xlabel(cat_var)
        axes[0].set_ylabel(num_var)
        
        # Violin plot
        sns.violinplot(x=cat_var, y=num_var, data=df, ax=axes[1], palette='Set2')
        axes[1].set_title(f'Violin Plot: {num_var} by {cat_var}', fontweight='bold')
        axes[1].set_xlabel(cat_var)
        axes[1].set_ylabel(num_var)
        
        plt.tight_layout()
        st.pyplot(fig)

def perform_multivariate_analysis(df):
    """Perform comprehensive multivariate analysis"""
    st.markdown("## 🎯 Multivariate Analysis")
    st.markdown("*Analyzing relationships among three or more variables*")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Correlation Heatmap", "Pair Plot", "3D Scatter Plot", "PCA Analysis"]
    )
    
    if analysis_type == "Correlation Heatmap" and len(num_cols) >= 2:
        st.markdown("### Correlation Heatmap")
        selected_cols = st.multiselect("Select numerical columns", num_cols, default=num_cols[:5])
        
        if len(selected_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[selected_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                       square=True, ax=ax, cbar_kws={'label': 'Correlation'})
            ax.set_title('Correlation Matrix Heatmap', fontweight='bold', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
    
    elif analysis_type == "Pair Plot" and len(num_cols) >= 2:
        st.markdown("### Pair Plot")
        selected_cols = st.multiselect("Select numerical columns", num_cols, default=num_cols[:3])
        
        if len(selected_cols) >= 2:
            hue_col = st.selectbox("Select column for color coding (optional)", [None] + cat_cols)
            
            with st.spinner("Generating pair plot..."):
                fig = plt.figure(figsize=(12, 10))
                if hue_col and hue_col in df.columns:
                    pairplot_data = df[selected_cols + [hue_col]].dropna()
                else:
                    pairplot_data = df[selected_cols].dropna()
                
                # Create simplified pair plot
                n_vars = len(selected_cols)
                for i, col1 in enumerate(selected_cols):
                    for j, col2 in enumerate(selected_cols):
                        ax = plt.subplot(n_vars, n_vars, i * n_vars + j + 1)
                        if col1 == col2:
                            ax.hist(pairplot_data[col1], bins=20, alpha=0.7, color='steelblue')
                        else:
                            ax.scatter(pairplot_data[col1], pairplot_data[col2], alpha=0.5, s=20)
                        ax.set_xlabel(col1, fontsize=8)
                        ax.set_ylabel(col2, fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
    
    elif analysis_type == "3D Scatter Plot" and len(num_cols) >= 3:
        st.markdown("### 3D Scatter Plot")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_var = st.selectbox("X-axis variable", num_cols, key="3d_x")
        with col2:
            y_var = st.selectbox("Y-axis variable", num_cols, index=1 if len(num_cols) > 1 else 0, key="3d_y")
        with col3:
            z_var = st.selectbox("Z-axis variable", num_cols, index=2 if len(num_cols) > 2 else 0, key="3d_z")
        
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[x_var], df[y_var], df[z_var], alpha=0.6, c=df[x_var], cmap='viridis', s=50)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_zlabel(z_var)
        ax.set_title('3D Scatter Plot', fontweight='bold')
        st.pyplot(fig)
    
    elif analysis_type == "PCA Analysis" and len(num_cols) >= 2:
        st.markdown("### Principal Component Analysis (PCA)")
        selected_cols = st.multiselect("Select numerical columns for PCA", num_cols, default=num_cols[:5])
        n_components = st.slider("Number of components", 2, min(len(selected_cols), 10), 2)
        
        if len(selected_cols) >= 2:
            with st.spinner("Computing PCA..."):
                # Standardize data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[selected_cols].dropna())
                
                # PCA
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(scaled_data)
                
                # Variance explained
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                axes[0].bar(range(1, n_components + 1), pca.explained_variance_ratio_, 
                           color='steelblue', edgecolor='black', alpha=0.7)
                axes[0].set_xlabel('Principal Component')
                axes[0].set_ylabel('Explained Variance Ratio')
                axes[0].set_title('Variance Explained by Each PC', fontweight='bold')
                axes[0].grid(alpha=0.3, axis='y')
                
                axes[1].plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), 
                            marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=8)
                axes[1].set_xlabel('Number of Components')
                axes[1].set_ylabel('Cumulative Explained Variance')
                axes[1].set_title('Cumulative Variance Explained', fontweight='bold')
                axes[1].grid(alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown(f"**Explained Variance:**")
                var_df = pd.DataFrame({
                    'Component': [f'PC{i+1}' for i in range(n_components)],
                    'Variance Explained': pca.explained_variance_ratio_,
                    'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
                })
                st.dataframe(var_df, use_container_width=True)

def detect_and_handle_outliers_advanced(df):
    """Advanced outlier detection using multiple methods"""
    st.markdown("## 🎯 Advanced Outlier Detection & Treatment")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not num_cols:
        st.warning("No numerical columns found for outlier detection")
        return df
    
    detection_method = st.selectbox(
        "Select Outlier Detection Method",
        ["IQR Method", "Z-Score Method", "Isolation Forest", "Multiple Methods Comparison"]
    )
    
    if detection_method == "IQR Method":
        st.markdown("### IQR (Interquartile Range) Method")
        col = st.selectbox("Select column for outlier detection", num_cols, key="iqr_col")
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        st.markdown(f"""
        **IQR Statistics for {col}:**
        - Q1 (25th percentile): {Q1:.2f}
        - Q3 (75th percentile): {Q3:.2f}
        - IQR: {IQR:.2f}
        - Lower Bound: {lower_bound:.2f}
        - Upper Bound: {upper_bound:.2f}
        - Outliers Detected: {len(outliers)}
        """)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.boxplot(df[col].dropna())
        ax.set_ylabel(col)
        ax.set_title(f'Box Plot with Outlier Bounds - {col}', fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
    elif detection_method == "Z-Score Method":
        st.markdown("### Z-Score Method")
        col = st.selectbox("Select column for outlier detection", num_cols, key="zscore_col")
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.5)
        
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = df[z_scores > threshold]
        
        st.markdown(f"""
        **Z-Score Statistics for {col}:**
        - Threshold: {threshold}
        - Outliers Detected: {len(outliers)}
        - Percentage of Data: {(len(outliers)/len(df)*100):.2f}%
        """)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(range(len(df)), df[col], alpha=0.6, label='Normal', s=30)
        ax.scatter(outliers.index, outliers[col], color='red', label='Outliers', s=50)
        ax.set_xlabel('Index')
        ax.set_ylabel(col)
        ax.set_title(f'Outlier Detection using Z-Score - {col}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
    elif detection_method == "Isolation Forest":
        st.markdown("### Isolation Forest (Multivariate Method)")
        selected_cols = st.multiselect("Select columns for analysis", num_cols, default=num_cols[:3])
        contamination = st.slider("Contamination Rate (% of outliers)", 0.01, 0.2, 0.05, step=0.01)
        
        if len(selected_cols) >= 1:
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            df_temp = df[selected_cols].dropna()
            outlier_labels = iso_forest.fit_predict(df_temp)
            outlier_count = (outlier_labels == -1).sum()
            
            st.markdown(f"""
            **Isolation Forest Results for {', '.join(selected_cols)}:**
            - Anomalies Detected: {outlier_count}
            - Percentage of Data: {(outlier_count/len(df_temp)*100):.2f}%
            """)
            
            if len(selected_cols) == 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df_temp[selected_cols[0]], df_temp[selected_cols[1]], 
                          c=outlier_labels, cmap='coolwarm', s=50, alpha=0.6, edgecolor='black')
                ax.set_xlabel(selected_cols[0])
                ax.set_ylabel(selected_cols[1])
                ax.set_title('Isolation Forest Anomaly Detection', fontweight='bold')
                st.pyplot(fig)

def generate_eda_summary_report(df):
    """Generate comprehensive EDA summary report"""
    st.markdown("## 📋 EDA Summary Report")
    st.markdown("*Comprehensive exploratory data analysis overview*")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    report_sections = st.multiselect(
        "Select Report Sections",
        ["Data Overview", "Data Quality", "Statistical Summary", "Data Types", "Correlations", "Missing Data Analysis"],
        default=["Data Overview", "Data Quality", "Statistical Summary"]
    )
    
    if "Data Overview" in report_sections:
        st.markdown("### Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing %", f"{missing_pct:.2f}%")
    
    if "Data Quality" in report_sections:
        st.markdown("### Data Quality Metrics")
        quality_metrics = {
            'Column': [],
            'Non-Null Count': [],
            'Null Count': [],
            'Null %': [],
            'Data Type': [],
            'Unique Values': []
        }
        
        for col in df.columns:
            quality_metrics['Column'].append(col)
            quality_metrics['Non-Null Count'].append(df[col].notna().sum())
            quality_metrics['Null Count'].append(df[col].isnull().sum())
            quality_metrics['Null %'].append((df[col].isnull().sum() / len(df) * 100))
            quality_metrics['Data Type'].append(str(df[col].dtype))
            quality_metrics['Unique Values'].append(df[col].nunique())
        
        quality_df = pd.DataFrame(quality_metrics)
        st.dataframe(quality_df, use_container_width=True)
    
    if "Statistical Summary" in report_sections and num_cols:
        st.markdown("### Statistical Summary")
        st.dataframe(df[num_cols].describe().T, use_container_width=True)
    
    if "Data Types" in report_sections:
        st.markdown("### Data Types Distribution")
        dtype_counts = df.dtypes.value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        dtype_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_title('Data Types Distribution', fontweight='bold')
        ax.set_xlabel('Data Type')
        ax.set_ylabel('Count')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
    
    if "Correlations" in report_sections and len(num_cols) >= 2:
        st.markdown("### Correlation Analysis")
        corr_matrix = df[num_cols].corr()
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            st.markdown("**High Correlations (> 0.7):**")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            st.dataframe(high_corr_df, use_container_width=True)
        else:
            st.info("No correlations > 0.7 found")
    
    if "Missing Data Analysis" in report_sections:
        st.markdown("### Missing Data Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data[missing_data > 0].index,
                'Missing Count': missing_data[missing_data > 0].values,
                'Missing %': (missing_data[missing_data > 0].values / len(df) * 100)
            })
            st.dataframe(missing_df, use_container_width=True)
            
            # Visualize missing data
            fig, ax = plt.subplots(figsize=(10, 5))
            missing_df.set_index('Column')['Missing %'].plot(kind='barh', ax=ax, color='coral', edgecolor='black')
            ax.set_xlabel('Missing %')
            ax.set_title('Missing Data by Column', fontweight='bold')
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.success("✅ No missing data detected!")

# ============================================================================
# SIDEBAR - DATA SOURCE CONNECTOR
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

def handle_file_upload():
    """Handle CSV/Excel file uploads"""
    st.sidebar.subheader("Upload Files")
    uploaded_files = st.sidebar.file_uploader(
        "Select CSV or Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload one or more CSV/Excel files for analysis"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    df = load_csv(file)
                else:
                    df = load_excel(file)
                st.session_state.datasets[file.name] = df
                st.sidebar.success(f"✅ Loaded: {file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {file.name}: {str(e)}")

def handle_mysql_connection():
    """Handle MySQL database connection with credential management"""
    st.sidebar.subheader("MySQL Configuration")
    
    if CREDENTIAL_MANAGER_AVAILABLE:
        with st.sidebar.expander("🔐 Credential Profiles", expanded=False):
            render_credential_manager_ui("mysql")
            loaded_creds = get_loaded_credentials("mysql")
            if loaded_creds:
                host = loaded_creds.get('host', 'localhost')
                port = loaded_creds.get('port', '3306')
                user = loaded_creds.get('username', '')
                password = loaded_creds.get('password', '')
                database = loaded_creds.get('database', '')
                st.sidebar.success("✅ Credentials loaded from profile")
            else:
                host, port, user, password, database = None, None, None, None, None
    else:
        host, port, user, password, database = None, None, None, None, None
    
    with st.sidebar.expander("Connection Details", expanded=True):
        host = st.text_input("Host", value=host or "localhost", key="mysql_host")
        port = st.text_input("Port", value=port or "3306", key="mysql_port")
        user = st.text_input("Username", value=user or "", key="mysql_user")
        password = st.text_input("Password", value=password or "", type="password", key="mysql_pass")
        database = st.text_input("Database", value=database or "", key="mysql_db")
    
    if st.sidebar.button("🔍 Test & List Tables", key="mysql_test_list"):
        if all([host, user, password, database]):
            try:
                with st.spinner("Testing connection..."):
                    import pymysql
                    conn = pymysql.connect(
                        host=host,
                        user=user,
                        password=password,
                        db=database,
                        connect_timeout=10
                    )
                    with conn.cursor() as cursor:
                        cursor.execute("SHOW TABLES;")
                        tables = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    st.session_state.mysql_tables = tables
                    st.sidebar.success(f"✅ Connected! Found {len(tables)} tables")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
        else:
            st.sidebar.warning("Please fill all connection fields")
    
    if 'mysql_tables' in st.session_state and st.session_state.mysql_tables:
        selected_table = st.sidebar.selectbox(
            "Select Table",
            st.session_state.mysql_tables,
            key="mysql_table_select"
        )
        
        if st.sidebar.button("📥 Load Table", key="mysql_load"):
            try:
                query = f"SELECT * FROM {selected_table} LIMIT 10000"
                df = load_mysql(host, user, password, database, query)
                st.session_state.datasets[f"{database}.{selected_table}"] = df
                st.sidebar.success(f"✅ Loaded: {selected_table} ({len(df)} rows)")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    return host, user, password, database

def handle_postgres_connection():
    """Handle PostgreSQL database connection"""
    st.sidebar.subheader("PostgreSQL Configuration")
    
    if CREDENTIAL_MANAGER_AVAILABLE:
        with st.sidebar.expander("🔐 Credential Profiles", expanded=False):
            render_credential_manager_ui("postgres")
            loaded_creds = get_loaded_credentials("postgres")
            if loaded_creds:
                host = loaded_creds.get('host', 'localhost')
                port = loaded_creds.get('port', '5432')
                user = loaded_creds.get('username', '')
                password = loaded_creds.get('password', '')
                database = loaded_creds.get('database', '')
            else:
                host, port, user, password, database = None, None, None, None, None
    else:
        host, port, user, password, database = None, None, None, None, None
    
    with st.sidebar.expander("Connection Details", expanded=True):
        host = st.text_input("Host", value=host or "localhost", key="pg_host")
        port = st.text_input("Port", value=port or "5432", key="pg_port")
        user = st.text_input("Username", value=user or "", key="pg_user")
        password = st.text_input("Password", value=password or "", type="password", key="pg_pass")
        database = st.text_input("Database", value=database or "", key="pg_db")
    
    if st.sidebar.button("🔍 Test & List Tables", key="pg_test_list"):
        if all([host, user, password, database]):
            try:
                with st.spinner("Testing connection..."):
                    import psycopg2
                    conn = psycopg2.connect(
                        host=host,
                        user=user,
                        password=password,
                        dbname=database,
                        connect_timeout=10
                    )
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
                        tables = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    st.session_state.pg_tables = tables
                    st.sidebar.success(f"✅ Connected! Found {len(tables)} tables")
            except Exception as e:
                st.sidebar.error(f"❌ Connection failed: {str(e)}")
        else:
            st.sidebar.warning("Please fill all connection fields")
    
    if 'pg_tables' in st.session_state and st.session_state.pg_tables:
        selected_table = st.sidebar.selectbox(
            "Select Table",
            st.session_state.pg_tables,
            key="pg_table_select"
        )
        
        if st.sidebar.button("📥 Load Table", key="pg_load"):
            try:
                query = f"SELECT * FROM {selected_table} LIMIT 10000"
                df = load_postgres(host, user, password, database, query)
                st.session_state.datasets[f"{database}.{selected_table}"] = df
                st.sidebar.success(f"✅ Loaded: {selected_table} ({len(df)} rows)")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    return host, user, password, database

def handle_s3_connection():
    """Handle AWS S3 connection"""
    st.sidebar.subheader("AWS S3 Configuration")
    
    with st.sidebar.expander("S3 Credentials", expanded=True):
        bucket = st.text_input("Bucket Name", key="s3_bucket")
        aws_key = st.text_input("AWS Access Key ID", key="s3_key")
        aws_secret = st.text_input("AWS Secret Access Key", type="password", key="s3_secret")
    
    if st.sidebar.button("📂 List Objects", key="s3_list"):
        if all([bucket, aws_key, aws_secret]):
            try:
                with st.spinner("Loading S3 objects..."):
                    objects = list_s3_objects(bucket, aws_key, aws_secret)
                    st.session_state.s3_objects = objects
                    st.sidebar.success(f"Found {len(objects)} objects")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Please fill all fields")
    
    if 's3_objects' in st.session_state and st.session_state.s3_objects:
        selected_object = st.sidebar.selectbox(
            "Select Object",
            st.session_state.s3_objects,
            key="s3_object_select"
        )
        
        if st.sidebar.button("📥 Load Object", key="s3_load"):
            try:
                df = load_s3(bucket, selected_object, aws_key, aws_secret)
                st.session_state.datasets[f"s3://{bucket}/{selected_object}"] = df
                st.sidebar.success(f"✅ Loaded: {selected_object}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    return bucket, aws_key, aws_secret

def handle_mongodb_connection():
    """Handle MongoDB connection"""
    st.sidebar.subheader("MongoDB Configuration")
    
    with st.sidebar.expander("Connection Details", expanded=True):
        uri = st.text_input("MongoDB URI", "mongodb://localhost:27017/", key="mongo_uri")
        db_name = st.text_input("Database Name", key="mongo_db")
        collection = st.text_input("Collection Name", key="mongo_collection")
    
    if st.sidebar.button("📥 Load Collection", key="mongo_load"):
        if all([uri, db_name, collection]):
            try:
                df = load_mongodb(uri, db_name, collection)
                st.session_state.datasets[f"{db_name}.{collection}"] = df
                st.sidebar.success(f"✅ Loaded: {collection} ({len(df)} rows)")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Please fill all fields")

def handle_rest_api_connection():
    """Handle REST API connection with modal interface"""
    st.sidebar.subheader("REST API Configuration")
    
    # Single button to open API builder
    if st.sidebar.button("🌐 Open API Builder", key="api_open_builder", use_container_width=True):
        st.session_state.show_api_builder = True
    
    st.sidebar.markdown("---")
    
    # Quick GET Request section
    with st.sidebar.expander("Quick GET Request", expanded=False):
        url = st.text_input("API Endpoint URL", key="api_url_quick")
        
        if st.button("📥 Fetch Data (GET)", key="api_fetch_quick", use_container_width=True):
            if url:
                try:
                    if POSTMAN_UI_AVAILABLE:
                        df, metadata = load_rest_api(url)
                    else:
                        from autoeda.connectors.rest_api_connector import load_rest_api as basic_load
                        df = basic_load(url)
                        metadata = {}
                    
                    st.session_state.datasets[f"API: {url}"] = df
                    st.session_state.api_response_metadata[url] = metadata
                    st.sidebar.success(f"✅ Loaded {len(df)} records")
                except Exception as e:
                    st.sidebar.error(f"Error: {str(e)}")
            else:
                st.sidebar.warning("Please enter API URL")

# ============================================================================
# API BUILDER MODAL
# ============================================================================

@st.dialog("🌐 API Request Builder", width="large")
def show_api_builder_modal():
    """Display API builder in a modal dialog"""
    
    if POSTMAN_UI_AVAILABLE:
        try:
            api_config, execute_clicked, test_clicked = render_postman_style_api_interface("modal_api")
            
            # Handle API execution
            if execute_clicked and not test_clicked:
                if not api_config['url']:
                    st.error("Please enter an API URL")
                else:
                    try:
                        with st.spinner("Sending request..."):
                            df, metadata = load_rest_api(**api_config)
                            
                            # Save to datasets
                            dataset_name = f"API: {api_config['method']} {api_config['url']}"
                            st.session_state.datasets[dataset_name] = df
                            st.session_state.api_response_metadata[dataset_name] = metadata
                            
                            # Display response
                            st.success(f"✅ Request successful! Loaded {len(df)} records")
                            
                            # Response metadata
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Status Code", metadata.get('status_code', 'N/A'))
                            with col2:
                                st.metric("Response Time", f"{metadata.get('elapsed_time', 0):.2f}s")
                            with col3:
                                st.metric("Rows", len(df))
                            with col4:
                                st.metric("Columns", len(df.columns))
                            
                            # Preview data
                            st.markdown("### 📋 Response Data Preview")
                            st.dataframe(df.head(100), use_container_width=True)
                            
                            # Response headers
                            with st.expander("📄 Response Headers"):
                                st.json(metadata.get('headers', {}))
                            
                            # Close button
                            if st.button("✅ Done - Close Builder", type="primary", use_container_width=True):
                                st.session_state.show_api_builder = False
                                st.rerun()
                            
                    except Exception as e:
                        st.error(f"❌ Request failed: {str(e)}")
            
            elif test_clicked:
                if not api_config['url']:
                    st.error("Please enter an API URL")
                else:
                    try:
                        with st.spinner("Testing connection..."):
                            success, message, metadata = test_api_connection(
                                api_config['url'],
                                api_config['method'],
                                api_config['headers'],
                                api_config['auth_type'],
                                api_config['auth_credentials']
                            )
                            
                            if success:
                                st.success(message)
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Status Code", metadata.get('status_code', 'N/A'))
                                with col2:
                                    st.metric("Response Time", f"{metadata.get('response_time', 0):.3f}s")
                            else:
                                st.error(message)
                    except Exception as e:
                        st.error(f"❌ Connection test failed: {str(e)}")
        
        except Exception as e:
            st.error(f"API Builder Error: {str(e)}")
    else:
        st.warning("⚠️ Postman-style API UI not available. Install the required dependency.")

# ============================================================================
# WELCOME SCREEN
# ============================================================================

def render_welcome_screen():
    """Render modern welcome screen using Streamlit native components"""
    
    # Create a container with colored background using Streamlit
    
    # Main title and subtitle
    st.markdown(
        """
<div style="
    text-align: center; 
    padding: 3rem 2rem; 
    border-radius: 20px; 
    color: white; 
    margin-bottom: 2rem; 
    background: url('https://jmangroup.com/wp-content/uploads/2024/04/Group-688.svg') no-repeat center/cover;
">
    <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); color: #0f172a;">AutoEDA Platform</h1>
    <p style="font-size: 1.1rem; margin-bottom: 2rem; opacity: 0.9;">Transform raw data into actionable insights </p>
</div>

        """,
        unsafe_allow_html=True
    )
    
    # Statistics
    st.markdown("### 📊 Platform Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Data Sources", value="10+", delta="Files, DBs, APIs, Cloud")
    
    with col2:
        st.metric(label="Analysis Types", value="50+", delta="Statistical Methods")
    
    
    st.markdown("---")
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📊 Smart Analytics
        Advanced statistical analysis with univariate, bivariate, and multivariate techniques
        
        #### 🔗 Universal Connectivity
        Connect to databases, cloud storage, APIs, and file systems seamlessly
        """)
    
    with col2:
        st.markdown("""
        #### 🎯 Outlier Detection
        Multiple detection methods including IQR, Z-Score, and Isolation Forest
        
        #### 🛠️ Data Preprocessing
        Complete suite of cleaning, transformation, and feature engineering tools
        """)
    
    st.markdown("---")
    
    # Platform Capabilities
    st.markdown("### 🚀 Platform Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        - ✓ Real-time Visualization
        - ✓ PCA Analysis
        """)
    
    with col2:
        st.markdown("""
        - ✓ Correlation Analysis
        - ✓ Missing Data Treatment
        """)
    
    with col3:
        st.markdown("""
        - ✓ Feature Engineering
        - ✓ Custom SQL Queries
        """)
    
    with col4:
        st.markdown("""
        - ✓ Multi-Dataset Joins
        - ✓ Multiple Export Formats
        """)
    
    st.markdown("---")
    
    # Getting started section
    st.markdown("## 🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Connect Data Source
        - Choose from 10+ data sources
        - Files, databases, APIs, cloud storage
        - Secure credential management
        - Test connections before loading
        """)
    
    with col2:
        st.markdown("""
        ### 2️⃣ Explore & Analyze
        - Automatic data profiling
        - Interactive visualizations
        - Statistical summaries
        - Correlation analysis
        """)
    
    with col3:
        st.markdown("""
        ### 3️⃣ Process & Export
        - Clean and transform data
        - Handle missing values
        - Engineer features
        - Export in multiple formats
        """)
    
    st.markdown("---")
    
    # Quick tips
    with st.expander("💡 Quick Tips", expanded=False):
        st.markdown("""
        - **Start Simple**: Begin with file uploads to get familiar with the platform
        - **Use SQL Editor**: For databases, use custom SQL queries for precise data extraction
        - **Save Versions**: Create snapshots of your preprocessed data at different stages
        - **Join Datasets**: Combine multiple datasets for comprehensive analysis
        - **Export Early**: Download intermediate results to avoid data loss
        """)
    
    # Call to action
    st.info("👈 **Get Started:** Select a data source from the sidebar to begin your analysis journey!")

# ============================================================================
# ENHANCED DATA EXPLORATION
# ============================================================================

def render_data_exploration(df):
    """Render enhanced data exploration section"""
    st.markdown("## 📊 Exploratory Data Analysis")
    
    tab_names = ['📈 Overview', '📊 Univariate', '🔗 Bivariate', '🎯 Multivariate', 
                 '🎯 Outliers', '📋 Summary Report']
    tabs = st.tabs(tab_names)
    
    num_cols, cat_cols = analysis.categorical_numerical(df)
    
    # Tab 1: Overview
    with tabs[0]:
        st.subheader("Dataset Overview")
        analysis.display_dataset_overview(df, cat_cols, num_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Missing Values")
            analysis.display_missing_values(df)
        
        with col2:
            st.subheader("Data Types")
            analysis.display_data_types(df)
        
        st.subheader("Statistical Summary & Visualizations")
        analysis.display_statistics_visualization(df, cat_cols, num_cols)
        
        st.subheader("Column Search")
        analysis.search_column(df)
    
    # Tab 2: Univariate Analysis
    with tabs[1]:
        perform_univariate_analysis(df)
    
    # Tab 3: Bivariate Analysis
    with tabs[2]:
        perform_bivariate_analysis(df)
    
    # Tab 4: Multivariate Analysis
    with tabs[3]:
        perform_multivariate_analysis(df)
    
    # Tab 5: Outlier Detection
    with tabs[4]:
        detect_and_handle_outliers_advanced(df)
    
    # Tab 6: Summary Report
    with tabs[5]:
        generate_eda_summary_report(df)

def render_data_preprocessing(df):
    """Render data preprocessing section"""
    st.markdown("## 🛠️ Data Preprocessing")
    
    # Revert button
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        if st.button("↺ Revert to Original Dataset", type="secondary", key="revert_btn"):
            st.session_state.new_df = df.copy()
            st.success("Dataset reverted to original")
            st.rerun()
    
    with col2:
        st.info(f"Current: {st.session_state.new_df.shape[0]} rows × {st.session_state.new_df.shape[1]} columns")
    
    with col3:
        if st.button("💾 Save Version", key="save_version"):
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.datasets[f"Preprocessed_{timestamp}"] = st.session_state.new_df.copy()
            st.success("Version saved!")
    
    # Preprocessing options in tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗑️ Remove Columns",
        "🔧 Handle Missing Data",
        "🔤 Encode Categorical",
        "📏 Feature Scaling",
        "🎯 Handle Outliers",
        "➕ Feature Engineering"
    ])
    
    # Tab 1: Remove Columns
    with tab1:
        st.subheader("Remove Unwanted Columns")
        cols_to_remove = st.multiselect(
            "Select columns to remove",
            options=st.session_state.new_df.columns,
            help="Select one or more columns to remove from the dataset",
            key="cols_to_remove"
        )
        
        if st.button("Remove Columns", key="remove_cols"):
            if cols_to_remove:
                st.session_state.new_df = preprocessing.remove_selected_columns(
                    st.session_state.new_df, cols_to_remove
                )
                st.success(f"✅ Removed {len(cols_to_remove)} columns")
                st.dataframe(st.session_state.new_df.head())
                st.rerun()
    
    # Tab 2: Handle Missing Data
    with tab2:
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()
        
        if missing_count.any():
            st.warning(f"Found missing values in {(missing_count > 0).sum()} columns")
            
            # Display missing data visualization
            missing_df = pd.DataFrame({
                'Column': missing_count[missing_count > 0].index,
                'Missing Count': missing_count[missing_count > 0].values,
                'Percentage': (missing_count[missing_count > 0].values / len(st.session_state.new_df) * 100).round(2)
            }).sort_values('Missing Count', ascending=False)
            
            st.dataframe(missing_df, hide_index=True)
            
            option = st.radio(
                "Select method",
                ["Remove Rows", "Fill Missing Values (Numerical)", "Forward Fill", "Backward Fill"],
                horizontal=True,
                key="missing_data_method"
            )
            
            if option == "Remove Rows":
                cols = st.multiselect("Select columns", st.session_state.new_df.columns, key="remove_missing_cols")
                if st.button("Remove Rows with Missing Data", key="remove_missing_rows"):
                    st.session_state.new_df = preprocessing.remove_rows_with_missing_data(
                        st.session_state.new_df, cols
                    )
                    st.success("✅ Rows removed")
                    st.rerun()
            elif option == "Fill Missing Values (Numerical)":
                num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
                cols = st.multiselect("Select numerical columns", num_cols, key="fill_missing_cols")
                method = st.selectbox("Fill method", ["mean", "median", "mode"], key="fill_method")
                
                if st.button("Fill Missing Data", key="fill_missing_btn"):
                    st.session_state.new_df = preprocessing.fill_missing_data(
                        st.session_state.new_df, cols, method
                    )
                    st.success("✅ Missing data filled")
                    st.rerun()
            elif option == "Forward Fill":
                if st.button("Apply Forward Fill", key="ffill_btn"):
                    st.session_state.new_df = st.session_state.new_df.fillna(method='ffill')
                    st.success("✅ Forward fill applied")
                    st.rerun()
            else:  # Backward Fill
                if st.button("Apply Backward Fill", key="bfill_btn"):
                    st.session_state.new_df = st.session_state.new_df.fillna(method='bfill')
                    st.success("✅ Backward fill applied")
                    st.rerun()
            
            analysis.display_missing_values(st.session_state.new_df)
        else:
            st.success("✅ No missing values detected")
    
    # Tab 3: Encode Categorical
    with tab3:
        st.subheader("Encode Categorical Data")
        cat_cols = st.session_state.new_df.select_dtypes(include=['object']).columns
        
        if not cat_cols.empty:
            selected = st.multiselect("Select categorical columns", cat_cols, key="encode_cols")
            method = st.radio("Encoding Method", ['One Hot Encoding', 'Label Encoding'], horizontal=True, key="encode_method")
            
            if st.button("Apply Encoding", key="apply_encoding"):
                if selected:
                    if method == "One Hot Encoding":
                        st.session_state.new_df = preprocessing.one_hot_encode(
                            st.session_state.new_df, selected
                        )
                    else:
                        st.session_state.new_df = preprocessing.label_encode(
                            st.session_state.new_df, selected
                        )
                    st.success(f"✅ {method} applied")
                    st.dataframe(st.session_state.new_df.head())
                    st.rerun()
        else:
            st.info("No categorical columns found")
    
    # Tab 4: Feature Scaling
    with tab4:
        st.subheader("Feature Scaling")
        num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
        
        selected = st.multiselect("Select numerical columns", num_cols, key="scaling_cols")
        method = st.radio("Scaling Method", ['Standardization (Z-score)', 'Min-Max Scaling', 'Robust Scaling'], 
                         horizontal=True, key="scaling_method")
        
        if st.button("Apply Scaling", key="apply_scaling"):
            if selected:
                if method == "Standardization (Z-score)":
                    scaler = StandardScaler()
                    st.session_state.new_df[selected] = scaler.fit_transform(st.session_state.new_df[selected])
                elif method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    st.session_state.new_df[selected] = scaler.fit_transform(st.session_state.new_df[selected])
                else:
                    scaler = RobustScaler()
                    st.session_state.new_df[selected] = scaler.fit_transform(st.session_state.new_df[selected])
                
                st.success(f"✅ {method} applied")
                st.dataframe(st.session_state.new_df.head())
                st.rerun()
    
    # Tab 5: Handle Outliers
    with tab5:
        st.subheader("Identify and Handle Outliers")
        num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
        
        if not num_cols.empty:
            selected_col = st.selectbox("Select column", num_cols, key="outlier_col")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot(st.session_state.new_df[selected_col].dropna())
            ax.set_ylabel(selected_col)
            ax.set_title(f"Box Plot - {selected_col}")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            
            outliers = preprocessing.detect_outliers_zscore(st.session_state.new_df, selected_col)
            
            if outliers:
                st.warning(f"⚠️ Detected {len(outliers)} outliers using Z-score method (threshold=3)")
                method = st.radio("Handling Method", ["Remove Outliers", "Cap Outliers (Winsorize)"], horizontal=True, key="outlier_method")
                
                if st.button("Apply Outlier Handling", key="apply_outlier_handling"):
                    if method == "Remove Outliers":
                        st.session_state.new_df = preprocessing.remove_outliers(
                            st.session_state.new_df, selected_col, outliers
                        )
                    else:
                        st.session_state.new_df = preprocessing.transform_outliers(
                            st.session_state.new_df, selected_col, outliers
                        )
                    st.success("✅ Outliers handled")
                    st.rerun()
            else:
                st.success("✅ No outliers detected (Z-score threshold = 3)")
    
    # Tab 6: Feature Engineering
    with tab6:
        st.subheader("Feature Engineering")
        
        st.markdown("#### Create New Feature")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_type = st.selectbox(
                "Feature Type",
                ["Mathematical Operation", "Binning", "Date Features"],
                key="feature_type"
            )
        
        with col2:
            new_feature_name = st.text_input("New Feature Name", key="new_feature_name")
        
        if feature_type == "Mathematical Operation":
            st.markdown("**Mathematical Operation**")
            col1_math = st.selectbox("Column 1", st.session_state.new_df.select_dtypes(include=['number']).columns, key="math_col1")
            operation = st.selectbox("Operation", ["+", "-", "*", "/"], key="math_op")
            col2_math = st.selectbox("Column 2", st.session_state.new_df.select_dtypes(include=['number']).columns, key="math_col2")
            
            if st.button("Create Feature", key="create_math_feature"):
                if new_feature_name:
                    try:
                        if operation == "+":
                            st.session_state.new_df[new_feature_name] = st.session_state.new_df[col1_math] + st.session_state.new_df[col2_math]
                        elif operation == "-":
                            st.session_state.new_df[new_feature_name] = st.session_state.new_df[col1_math] - st.session_state.new_df[col2_math]
                        elif operation == "*":
                            st.session_state.new_df[new_feature_name] = st.session_state.new_df[col1_math] * st.session_state.new_df[col2_math]
                        elif operation == "/":
                            st.session_state.new_df[new_feature_name] = st.session_state.new_df[col1_math] / st.session_state.new_df[col2_math]
                        
                        st.success(f"✅ Created: {new_feature_name}")
                        st.dataframe(st.session_state.new_df[[col1_math, col2_math, new_feature_name]].head())
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("Please enter a feature name")
        
        elif feature_type == "Binning":
            st.markdown("**Binning (Discretization)**")
            bin_col = st.selectbox("Column to Bin", st.session_state.new_df.select_dtypes(include=['number']).columns, key="bin_col")
            n_bins = st.slider("Number of Bins", 2, 10, 5, key="n_bins")
            
            if st.button("Create Binned Feature", key="create_bin_feature"):
                if new_feature_name:
                    st.session_state.new_df[new_feature_name] = pd.cut(st.session_state.new_df[bin_col], bins=n_bins, labels=False)
                    st.success(f"✅ Created: {new_feature_name}")
                    st.rerun()
                else:
                    st.warning("Please enter a feature name")
        
        elif feature_type == "Date Features":
            st.markdown("**Date Feature Extraction**")
            date_cols = st.session_state.new_df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) == 0:
                st.info("No datetime columns found. Convert columns to datetime first.")
            else:
                date_col = st.selectbox("Date Column", date_cols, key="date_col")
                features = st.multiselect(
                    "Extract Features",
                    ["Year", "Month", "Day", "Day of Week", "Quarter", "Week of Year"],
                    key="date_features"
                )
                
                if st.button("Extract Date Features", key="extract_date"):
                    for feature in features:
                        if feature == "Year":
                            st.session_state.new_df[f"{date_col}_year"] = st.session_state.new_df[date_col].dt.year
                        elif feature == "Month":
                            st.session_state.new_df[f"{date_col}_month"] = st.session_state.new_df[date_col].dt.month
                        elif feature == "Day":
                            st.session_state.new_df[f"{date_col}_day"] = st.session_state.new_df[date_col].dt.day
                        elif feature == "Day of Week":
                            st.session_state.new_df[f"{date_col}_dayofweek"] = st.session_state.new_df[date_col].dt.dayofweek
                        elif feature == "Quarter":
                            st.session_state.new_df[f"{date_col}_quarter"] = st.session_state.new_df[date_col].dt.quarter
                        elif feature == "Week of Year":
                            st.session_state.new_df[f"{date_col}_week"] = st.session_state.new_df[date_col].dt.isocalendar().week
                    
                    st.success(f"✅ Extracted {len(features)} date features")
                    st.rerun()
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Download Preprocessed Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = st.session_state.new_df.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name="preprocessed_data.csv",
            mime="text/csv",
            key="download_csv",
            use_container_width=True
        )
    
    with col2:
        try:
            from io import BytesIO
            buffer = BytesIO()
            st.session_state.new_df.to_excel(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="📊 Download as Excel",
                data=buffer,
                file_name="preprocessed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel",
                use_container_width=True
            )
        except:
            st.info("Install openpyxl for Excel export")

def render_dataset_join_section():
    """Render dataset join functionality"""
    if len(st.session_state.datasets) >= 2:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔗 Join Datasets")
        
        dataset_names = list(st.session_state.datasets.keys())
        
        with st.sidebar.expander("Join Configuration"):
            left_ds = st.selectbox("Left Dataset", dataset_names, key="join_left")
            right_ds = st.selectbox("Right Dataset", dataset_names, index=1 if len(dataset_names) > 1 else 0, key="join_right")
            join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"], key="join_type")
            
            df_left = st.session_state.datasets[left_ds]
            df_right = st.session_state.datasets[right_ds]
            
            left_col = st.selectbox("Left Join Column", df_left.columns, key="join_left_col")
            right_col = st.selectbox("Right Join Column", df_right.columns, key="join_right_col")
            
            if st.button("⚡ Join Datasets", key="join_execute"):
                try:
                    joined_df = df_left.merge(
                        df_right,
                        left_on=left_col,
                        right_on=right_col,
                        how=join_type
                    )
                    joined_name = f"Joined_{left_ds}_{right_ds}"
                    st.session_state.datasets[joined_name] = joined_df
                    st.sidebar.success(f"✅ Created: {joined_name} ({len(joined_df)} rows)")
                except Exception as e:
                    st.sidebar.error(f"Join error: {str(e)}")

def render_sql_editor(data_source, connection_params):
    """Render SQL editor for database sources"""
    db_sources = ["MySQL", "PostgreSQL", "SQLite", "MSSQL"]
    
    if data_source in db_sources:
        st.markdown("### 💻 Custom SQL Query Editor")
        st.markdown("*Execute custom queries to extract and transform data*")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            default_query = "SELECT * FROM your_table LIMIT 1000"
            query = st_ace(
                default_query,
                language="sql",
                theme="monokai",
                key=f"sql_editor_{data_source}_{st.session_state.session_id}",
                height=200
            )
        
        with col2:
            result_name = st.text_input("Result Name", f"{data_source}_Query_Result", key="sql_result_name")
            execute_btn = st.button("▶️ Execute Query", type="primary", key="sql_execute")
        
        if execute_btn:
            if not query.strip().lower().startswith("select"):
                st.warning("⚠️ Only SELECT queries are allowed for data safety")
            else:
                try:
                    with st.spinner("Executing query..."):
                        if data_source == "MySQL":
                            df = load_mysql(*connection_params, query)
                        elif data_source == "PostgreSQL":
                            df = load_postgres(*connection_params, query)
                        elif data_source == "SQLite":
                            df = load_sqlite(connection_params[0], query)
                        elif data_source == "MSSQL":
                            df = load_mssql(*connection_params, query)

                        st.session_state.datasets[result_name] = df
                        st.success(f"✅ Query executed! {len(df)} rows loaded as '{result_name}'")

                        if isinstance(df, pd.DataFrame):
                            if df.shape[1] == 1 and df.shape[0] == 1:
                                val = df.iloc[0, 0]
                                st.info(f"Result: {val}")
                                st.dataframe(df)
                            elif df.shape[0] == 1:
                                st.dataframe(df)
                            else:
                                st.dataframe(df.head(10))
                                st.caption(f"Showing first 10 rows of {len(df)} total rows.")
                        else:
                            st.write(df)
                except Exception as e:
                    st.error(f"❌ Query execution error: {str(e)}")

# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================

def main():
    """Main application entry point"""
    
    # Show API builder modal if requested
    if st.session_state.get('show_api_builder', False):
        show_api_builder_modal()
        # Reset the flag after showing
        if not st.session_state.get('show_api_builder', False):
            st.rerun()
    
    # Render sidebar and get data source
    data_source = render_sidebar()
    
    # Handle different data sources
    connection_params = None
    
    if data_source == "CSV/Excel File":
        handle_file_upload()
    elif data_source == "MySQL":
        connection_params = handle_mysql_connection()
    elif data_source == "PostgreSQL":
        connection_params = handle_postgres_connection()
    elif data_source == "AWS S3":
        connection_params = handle_s3_connection()
    elif data_source == "MongoDB":
        handle_mongodb_connection()
    elif data_source == "REST API":
        handle_rest_api_connection()
    
    # Dataset join section
    render_dataset_join_section()
    
    # SQL Editor for database sources
    if connection_params:
        render_sql_editor(data_source, connection_params)
        
    # Main content area - Dataset selection and analysis
    if st.session_state.datasets:
        st.markdown("## 📋 Dataset Selection")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_dataset = st.selectbox(
                "Select dataset for analysis",
                list(st.session_state.datasets.keys()),
                help="Choose a loaded dataset to explore and preprocess",
                key="dataset_selector"
            )
        
        with col2:
            if st.button("🗑️ Remove Dataset", key="remove_dataset"):
                del st.session_state.datasets[selected_dataset]
                st.success("Dataset removed")
                st.rerun()
        
        df = st.session_state.datasets[selected_dataset]
        
        # Dataset info card
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory", f"{memory_mb:.2f} MB")
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing %", f"{missing_pct:.2f}%")
        
        # Initialize working dataset
        if st.session_state.new_df is None or st.session_state.source_dataset != selected_dataset:
            st.session_state.new_df = df.copy()
            st.session_state.source_dataset = selected_dataset
        
        # Main menu
        selected_menu = option_menu(
            menu_title=None,
            options=['Data Exploration', 'Data Preprocessing'],
            icons=['bar-chart-fill', 'tools'],
            orientation='horizontal',
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "padding": "10px 20px",
                },
                "nav-link-selected": {"background-color": "#1f77b4"},
            },
            key="main_menu"
        )
        
        # Render selected section
        if selected_menu == 'Data Exploration':
            render_data_exploration(df)
        else:
            render_data_preprocessing(df)
    else:
        # Welcome screen - NEW IMPROVED UI
        render_welcome_screen()

if __name__ == "__main__":
    main()