"""
AutoEDA - Automated Exploratory Data Analysis Platform
A production-ready internal tool for streamlining EDA across DS projects
"""

import streamlit as st
import pandas as pd
import base64
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
import seaborn as sns
import uuid

# Import internal modules
import data_analysis_functions as analysis
import data_preprocessing_function as preprocessing

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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
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

init_session_state()

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
    """Handle MySQL database connection"""
    st.sidebar.subheader("MySQL Configuration")
    
    with st.sidebar.expander("Connection Details", expanded=True):
        host = st.text_input("Host", "localhost", key="mysql_host")
        port = st.text_input("Port", "3306", key="mysql_port")
        user = st.text_input("Username", key="mysql_user")
        password = st.text_input("Password", type="password", key="mysql_pass")
        database = st.text_input("Database", key="mysql_db")
    
    if st.sidebar.button("🔍 List Tables", key="mysql_list"):
        if all([host, user, password, database]):
            try:
                import pymysql
                conn = pymysql.connect(host=host, user=user, password=password, db=database)
                with conn.cursor() as cursor:
                    cursor.execute("SHOW TABLES;")
                    tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                st.session_state.mysql_tables = tables
                st.sidebar.success(f"Found {len(tables)} tables")
            except Exception as e:
                st.sidebar.error(f"Connection error: {str(e)}")
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
                query = f"SELECT * FROM {selected_table} LIMIT 1000"
                df = load_mysql(host, user, password, database, query)
                st.session_state.datasets[f"{database}.{selected_table}"] = df
                st.sidebar.success(f"✅ Loaded: {selected_table}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    return host, user, password, database

def handle_postgres_connection():
    """Handle PostgreSQL database connection"""
    st.sidebar.subheader("PostgreSQL Configuration")
    
    with st.sidebar.expander("Connection Details", expanded=True):
        host = st.text_input("Host", "localhost", key="pg_host")
        port = st.text_input("Port", "5432", key="pg_port")
        user = st.text_input("Username", key="pg_user")
        password = st.text_input("Password", type="password", key="pg_pass")
        database = st.text_input("Database", key="pg_db")
    
    if st.sidebar.button("🔍 List Tables", key="pg_list"):
        if all([host, user, password, database]):
            try:
                import psycopg2
                conn = psycopg2.connect(host=host, user=user, password=password, dbname=database)
                with conn.cursor() as cursor:
                    cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public';")
                    tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                st.session_state.pg_tables = tables
                st.sidebar.success(f"Found {len(tables)} tables")
            except Exception as e:
                st.sidebar.error(f"Connection error: {str(e)}")
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
                query = f"SELECT * FROM {selected_table} LIMIT 1000"
                df = load_postgres(host, user, password, database, query)
                st.session_state.datasets[f"{database}.{selected_table}"] = df
                st.sidebar.success(f"✅ Loaded: {selected_table}")
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
                st.sidebar.success(f"✅ Loaded: {collection}")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Please fill all fields")

def handle_rest_api_connection():
    """Handle REST API connection"""
    st.sidebar.subheader("REST API Configuration")
    
    url = st.sidebar.text_input("API Endpoint URL", key="api_url")
    
    if st.sidebar.button("📥 Fetch Data", key="api_fetch"):
        if url:
            try:
                df = load_rest_api(url)
                st.session_state.datasets[f"API: {url}"] = df
                st.sidebar.success("✅ Data loaded from API")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
        else:
            st.sidebar.warning("Please enter API URL")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

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
                    st.sidebar.success(f"✅ Created: {joined_name}")
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

                        # Show result for any SELECT, including count(*)
                        if isinstance(df, pd.DataFrame):
                            if df.shape[1] == 1 and df.shape[0] == 1:
                                # Single value (e.g., count(*))
                                val = df.iloc[0, 0]
                                st.info(f"Result: {val}")
                                st.dataframe(df)
                            elif df.shape[0] == 1:
                                st.dataframe(df)
                            else:
                                st.dataframe(df.head(10))
                                st.caption(f"Showing first 10 rows. Download for full result.")
                        else:
                            st.write(df)
                except Exception as e:
                    st.error(f"❌ Query execution error: {str(e)}")

def render_data_exploration(df):
    """Render data exploration section"""
    st.markdown("## 📊 Data Exploration & Analysis")
    
    tab1, tab2 = st.tabs(['📈 Dataset Overview', '🔍 Detailed Analysis'])
    
    num_cols, cat_cols = analysis.categorical_numerical(df)
    
    with tab1:
        st.subheader("Dataset Preview")
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
    
    with tab2:
        if num_cols:
            analysis.display_individual_feature_distribution(df, num_cols)
        
        if len(num_cols) >= 2:
            st.subheader("Scatter Plot Analysis")
            analysis.display_scatter_plot_of_two_numeric_features(df, num_cols)
        
        if cat_cols:
            st.subheader("Categorical Variable Analysis")
            analysis.categorical_variable_analysis(df, cat_cols)
        
        if num_cols:
            st.subheader("Numerical Feature Exploration")
            analysis.feature_exploration_numerical_variables(df, num_cols)
        
        if num_cols and cat_cols:
            st.subheader("Categorical vs Numerical Analysis")
            analysis.categorical_numerical_variable_analysis(df, cat_cols, num_cols)

def render_data_preprocessing(df):
    """Render data preprocessing section"""
    st.markdown("## 🛠️ Data Preprocessing")
    
    # Revert button
    if st.button("↺ Revert to Original Dataset", type="secondary", key="revert_btn"):
        st.session_state.new_df = df.copy()
        st.success("Dataset reverted to original")
        st.rerun()
    
    # Show current dataset shape
    st.info(f"Current dataset: {st.session_state.new_df.shape[0]} rows × {st.session_state.new_df.shape[1]} columns")
    
    # Preprocessing options in tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗑️ Remove Columns",
        "🔧 Handle Missing Data",
        "🔤 Encode Categorical",
        "📏 Feature Scaling",
        "🎯 Handle Outliers"
    ])
    
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
    
    with tab2:
        st.subheader("Handle Missing Data")
        missing_count = st.session_state.new_df.isnull().sum()
        
        if missing_count.any():
            st.warning(f"Found missing values in {(missing_count > 0).sum()} columns")
            
            option = st.radio(
                "Select method",
                ["Remove Rows", "Fill Missing Values (Numerical)"],
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
            else:
                num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
                cols = st.multiselect("Select numerical columns", num_cols, key="fill_missing_cols")
                method = st.selectbox("Fill method", ["mean", "median", "mode"], key="fill_method")
                
                if st.button("Fill Missing Data", key="fill_missing_btn"):
                    st.session_state.new_df = preprocessing.fill_missing_data(
                        st.session_state.new_df, cols, method
                    )
                    st.success("✅ Missing data filled")
            
            analysis.display_missing_values(st.session_state.new_df)
        else:
            st.success("✅ No missing values detected")
    
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
        else:
            st.info("No categorical columns found")
    
    with tab4:
        st.subheader("Feature Scaling")
        num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
        
        selected = st.multiselect("Select numerical columns", num_cols, key="scaling_cols")
        method = st.radio("Scaling Method", ['Standardization', 'Min-Max Scaling'], horizontal=True, key="scaling_method")
        
        if st.button("Apply Scaling", key="apply_scaling"):
            if selected:
                if method == "Standardization":
                    st.session_state.new_df = preprocessing.standard_scale(
                        st.session_state.new_df, selected
                    )
                else:
                    st.session_state.new_df = preprocessing.min_max_scale(
                        st.session_state.new_df, selected
                    )
                st.success(f"✅ {method} applied")
                st.dataframe(st.session_state.new_df.head())
    
    with tab5:
        st.subheader("Identify and Handle Outliers")
        num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
        
        if not num_cols.empty:
            selected_col = st.selectbox("Select column", num_cols, key="outlier_col")
            
            fig = sns.boxplot(data=st.session_state.new_df, x=selected_col)
            st.pyplot(fig.figure)
            
            outliers = preprocessing.detect_outliers_zscore(st.session_state.new_df, selected_col)
            
            if outliers:
                st.warning(f"⚠️ Detected {len(outliers)} outliers")
                method = st.radio("Handling Method", ["Remove Outliers", "Transform Outliers"], horizontal=True, key="outlier_method")
                
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
            else:
                st.success("✅ No outliers detected")
    
    # Download section
    st.markdown("---")
    st.subheader("📥 Download Preprocessed Data")
    
    csv = st.session_state.new_df.to_csv(index=False)
    st.download_button(
        label="📄 Download as CSV",
        data=csv,
        file_name="preprocessed_data.csv",
        mime="text/csv",
        key="download_csv"
    )

# ============================================================================
# MAIN APPLICATION FLOW
# ============================================================================

def main():
    """Main application entry point"""
    
    # Header
    st.markdown('<p class="main-header">AutoEDA Platform</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Automated Exploratory Data Analysis for DS Projects</p>', unsafe_allow_html=True)
    
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
    
    st.markdown("---")
    
    # Main content area - Dataset selection and analysis
    if st.session_state.datasets:
        st.markdown("## 📋 Dataset Selection")
        
        selected_dataset = st.selectbox(
            "Select dataset for analysis",
            list(st.session_state.datasets.keys()),
            help="Choose a loaded dataset to explore and preprocess",
            key="dataset_selector"
        )
        
        df = st.session_state.datasets[selected_dataset]
        
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
                "container": {"padding": "0!important"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px"},
            },
            key="main_menu"
        )
        
        st.markdown("---")
        
        # Render selected section
        if selected_menu == 'Data Exploration':
            render_data_exploration(df)
        else:
            render_data_preprocessing(df)
    else:
        st.info("👈 Please connect to a data source and load data from the sidebar")
        
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Select a data source** from the sidebar (Files, Databases, Cloud Storage, or APIs)
        2. **Configure connection** parameters as needed
        3. **Load your data** using the appropriate buttons
        4. **Select a dataset** from the dropdown to begin analysis
        5. **Explore and preprocess** your data using the menu options
        """)

if __name__ == "__main__":
    main()