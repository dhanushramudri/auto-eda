import streamlit as st
import pandas as pd
import base64
from streamlit_option_menu import option_menu
import seaborn as sns
import data_analysis_functions as function
import data_preprocessing_function as preprocessing_function

st.set_page_config(page_icon="✨", page_title="AutoEDA")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("AutoEDA")

# File uploader - Multiple files
uploaded_files = st.sidebar.file_uploader("Upload Multiple CSV Files", type=["csv", "xls"], accept_multiple_files=True)

if uploaded_files:
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}

    for file in uploaded_files:
        df = function.load_data(file)
        st.session_state.datasets[file.name] = df

# Dataset Join Section
st.sidebar.subheader("Join Datasets")
if 'datasets' in st.session_state and len(st.session_state.datasets) >= 2:
    dataset_names = list(st.session_state.datasets.keys())

    left_dataset_name = st.sidebar.selectbox("Select Left Dataset", dataset_names)
    right_dataset_name = st.sidebar.selectbox("Select Right Dataset", dataset_names, index=1)

    join_type = st.sidebar.selectbox("Join Type", ["inner", "left", "right", "outer"])

    df_left = st.session_state.datasets[left_dataset_name]
    df_right = st.session_state.datasets[right_dataset_name]

    left_join_column = st.sidebar.selectbox("Left Join Column", df_left.columns)
    right_join_column = st.sidebar.selectbox("Right Join Column", df_right.columns)

    if st.sidebar.button("Join Datasets"):
        if left_join_column in df_left.columns and right_join_column in df_right.columns:
            joined_df = df_left.merge(df_right, left_on=left_join_column, right_on=right_join_column, how=join_type)
            joined_name = f"Joined_{left_dataset_name}_{right_dataset_name}"
            st.session_state.datasets[joined_name] = joined_df
            st.success(f"Datasets joined and saved as: {joined_name}")
        else:
            st.sidebar.error("Join columns not found in selected datasets.")


# Select dataset for EDA
df = None
if 'datasets' in st.session_state and st.session_state.datasets:
    selected_dataset_name = st.sidebar.selectbox("Choose Dataset for EDA", list(st.session_state.datasets.keys()))
    df = st.session_state.datasets[selected_dataset_name]

    if 'new_df' not in st.session_state or st.session_state.source_dataset != selected_dataset_name:
        st.session_state.new_df = df.copy()
        st.session_state.source_dataset = selected_dataset_name

# Menu
selected = option_menu(
    menu_title=None,
    options=['Data Exploration', 'Data Preprocessing'],
    icons=['bar-chart-fill', 'hammer'],
    orientation='horizontal'
)

# ---- Data Exploration ----
if df is not None and selected == 'Data Exploration':
    tab1, tab2 = st.tabs(['📊 Dataset Overview', "🔎 Data Exploration"])

    num_columns, cat_columns = function.categorical_numerical(df)

    with tab1:
        st.subheader("1. Dataset Preview")
        function.display_dataset_overview(df, cat_columns, num_columns)

        st.subheader("3. Missing Values")
        function.display_missing_values(df)

        st.subheader("4. Data Statistics and Visualization")
        function.display_statistics_visualization(df, cat_columns, num_columns)

        st.subheader("5. Data Types")
        function.display_data_types(df)

        st.subheader("Search for a specific column or datatype")
        function.search_column(df)

    with tab2:
        function.display_individual_feature_distribution(df, num_columns)

        st.subheader("Scatter Plot")
        function.display_scatter_plot_of_two_numeric_features(df, num_columns)

        if cat_columns:
            st.subheader("Categorical Variable Analysis")
            function.categorical_variable_analysis(df, cat_columns)

        st.subheader("Feature Exploration of Numerical Variables")
        if num_columns:
            function.feature_exploration_numerical_variables(df, num_columns)

        st.subheader("Categorical and Numerical Variable Analysis")
        if num_columns and cat_columns:
            function.categorical_numerical_variable_analysis(df, cat_columns, num_columns)

# ---- Data Preprocessing ----
if df is not None and selected == 'Data Preprocessing':
    revert = st.button("Revert to Original Dataset", key="revert_button")
    if revert:
        st.session_state.new_df = df.copy()

    st.subheader("Remove Unwanted Columns")
    columns_to_remove = st.multiselect(label='Select Columns to Remove', options=st.session_state.new_df.columns)
    if st.button("Remove Selected Columns"):
        if columns_to_remove:
            st.session_state.new_df = preprocessing_function.remove_selected_columns(st.session_state.new_df, columns_to_remove)
            st.success("Selected Columns Removed")

    st.dataframe(st.session_state.new_df)

    st.subheader("Handle Missing Data")
    missing_count = st.session_state.new_df.isnull().sum()
    if missing_count.any():
        selected_missing_option = st.selectbox(
            "Select how to handle missing data:",
            ["Remove Rows in Selected Columns", "Fill Missing Data in Selected Columns (Numerical Only)"]
        )
        if selected_missing_option == "Remove Rows in Selected Columns":
            cols = st.multiselect("Select columns", options=st.session_state.new_df.columns)
            if st.button("Remove Rows with Missing Data"):
                st.session_state.new_df = preprocessing_function.remove_rows_with_missing_data(st.session_state.new_df, cols)
                st.success("Rows removed")
        elif selected_missing_option == "Fill Missing Data in Selected Columns (Numerical Only)":
            cols = st.multiselect("Select numerical columns", options=st.session_state.new_df.select_dtypes(include=['number']).columns)
            fill_method = st.selectbox("Fill method", ["mean", "median", "mode"])
            if st.button("Fill Missing Data"):
                st.session_state.new_df = preprocessing_function.fill_missing_data(st.session_state.new_df, cols, fill_method)
                st.success("Missing data filled")
        function.display_missing_values(st.session_state.new_df)
    else:
        st.info("No missing values")

    st.subheader("Encode Categorical Data")
    cat_cols = st.session_state.new_df.select_dtypes(include=['object']).columns
    if not cat_cols.empty:
        selected = st.multiselect("Select Columns", cat_cols)
        encoding_method = st.selectbox("Encoding Method", ['One Hot Encoding', 'Label Encoding'])
        if st.button("Apply Encoding"):
            if encoding_method == "One Hot Encoding":
                st.session_state.new_df = preprocessing_function.one_hot_encode(st.session_state.new_df, selected)
                st.success("One Hot Encoding Applied")
            elif encoding_method == "Label Encoding":
                st.session_state.new_df = preprocessing_function.label_encode(st.session_state.new_df, selected)
                st.success("Label Encoding Applied")
        st.dataframe(st.session_state.new_df)

    st.subheader("Feature Scaling")
    num_cols = st.session_state.new_df.select_dtypes(include=['number']).columns
    selected = st.multiselect("Select Columns", num_cols)
    scaling_method = st.selectbox("Scaling Method", ['Standardization', 'Min-Max Scaling'])
    if st.button("Apply Scaling"):
        if scaling_method == "Standardization":
            st.session_state.new_df = preprocessing_function.standard_scale(st.session_state.new_df, selected)
        elif scaling_method == "Min-Max Scaling":
            st.session_state.new_df = preprocessing_function.min_max_scale(st.session_state.new_df, selected)
        st.success("Scaling Applied")
    st.dataframe(st.session_state.new_df)

    st.subheader("Identify and Handle Outliers")
    selected_column = st.selectbox("Select Numeric Column", num_cols)
    fig = sns.boxplot(data=st.session_state.new_df, x=selected_column)
    st.pyplot(fig.figure)

    outliers = preprocessing_function.detect_outliers_zscore(st.session_state.new_df, selected_column)
    if outliers:
        st.warning("Outliers Detected")
        st.write(outliers)
    else:
        st.info("No outliers detected")

    outlier_method = st.selectbox("Handling Method", ["Remove Outliers", "Transform Outliers"])
    if st.button("Apply Outlier Handling"):
        if outlier_method == "Remove Outliers":
            st.session_state.new_df = preprocessing_function.remove_outliers(st.session_state.new_df, selected_column, outliers)
        elif outlier_method == "Transform Outliers":
            st.session_state.new_df = preprocessing_function.transform_outliers(st.session_state.new_df, selected_column, outliers)
        st.success("Outliers Handled")
    st.dataframe(st.session_state.new_df)

    # Download button
    if st.session_state.new_df is not None:
        csv = st.session_state.new_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        st.markdown(f'<a href="{href}" download="preprocessed_data.csv"><button>Download Preprocessed Data</button></a>', unsafe_allow_html=True)
