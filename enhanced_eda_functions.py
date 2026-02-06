"""
Enhanced EDA Analysis Functions
Advanced features for comprehensive exploratory data analysis
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from typing import List, Tuple, Dict, Any


def generate_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data quality report.
    
    Returns:
        Dictionary containing quality metrics
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicate_rows': df.duplicated().sum(),
        'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
    }
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    report['columns_with_missing'] = (missing_data > 0).sum()
    report['total_missing_values'] = missing_data.sum()
    report['missing_percentage'] = (report['total_missing_values'] / (len(df) * len(df.columns))) * 100
    
    # Data types
    report['numeric_columns'] = len(df.select_dtypes(include=[np.number]).columns)
    report['categorical_columns'] = len(df.select_dtypes(include=['object']).columns)
    report['datetime_columns'] = len(df.select_dtypes(include=['datetime64']).columns)
    
    # Cardinality analysis
    high_cardinality_cols = []
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95:
            high_cardinality_cols.append(col)
    report['high_cardinality_columns'] = high_cardinality_cols
    
    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    report['constant_columns'] = constant_cols
    
    return report


def display_data_quality_dashboard(df: pd.DataFrame):
    """Display interactive data quality dashboard"""
    st.markdown("### 📊 Data Quality Dashboard")
    
    report = generate_data_quality_report(df)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{report['total_rows']:,}")
        st.metric("Total Columns", report['total_columns'])
    
    with col2:
        st.metric("Memory Usage", f"{report['memory_usage_mb']:.2f} MB")
        st.metric("Duplicate Rows", f"{report['duplicate_rows']:,}")
    
    with col3:
        st.metric("Missing Values", f"{report['total_missing_values']:,}")
        st.metric("Missing %", f"{report['missing_percentage']:.2f}%")
    
    with col4:
        st.metric("Numeric Cols", report['numeric_columns'])
        st.metric("Categorical Cols", report['categorical_columns'])
    
    # Quality alerts
    st.markdown("#### ⚠️ Quality Alerts")
    
    alerts = []
    if report['duplicate_percentage'] > 5:
        alerts.append(f"🔴 High duplicate rate: {report['duplicate_percentage']:.2f}%")
    if report['missing_percentage'] > 10:
        alerts.append(f"🔴 High missing data: {report['missing_percentage']:.2f}%")
    if report['constant_columns']:
        alerts.append(f"🟡 {len(report['constant_columns'])} constant columns detected")
    if report['high_cardinality_columns']:
        alerts.append(f"🟡 {len(report['high_cardinality_columns'])} high cardinality columns")
    
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ No major quality issues detected")
    
    # Detailed breakdowns
    with st.expander("📋 Detailed Breakdown"):
        if report['constant_columns']:
            st.write("**Constant Columns:**", ", ".join(report['constant_columns']))
        if report['high_cardinality_columns']:
            st.write("**High Cardinality Columns:**", ", ".join(report['high_cardinality_columns']))


def display_correlation_analysis(df: pd.DataFrame, num_cols: List[str]):
    """Advanced correlation analysis with multiple methods"""
    st.markdown("### 🔗 Correlation Analysis")
    
    if len(num_cols) < 2:
        st.info("Need at least 2 numerical columns for correlation analysis")
        return
    
    # Correlation method selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        corr_method = st.selectbox(
            "Correlation Method",
            ["Pearson", "Spearman", "Kendall"],
            help="Pearson: linear relationships, Spearman/Kendall: monotonic relationships"
        )
        
        threshold = st.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Show correlations above this threshold"
        )
    
    with col2:
        # Calculate correlation matrix
        method_map = {"Pearson": "pearson", "Spearman": "spearman", "Kendall": "kendall"}
        corr_matrix = df[num_cols].corr(method=method_map[corr_method])
        
        # Create interactive heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title=f"{corr_method} Correlation Heatmap"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Strong correlations table
    st.markdown("#### 🎯 Strong Correlations")
    
    # Get upper triangle of correlation matrix
    upper_triangle = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
    strong_corrs = []
    
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                strong_corrs.append({
                    'Feature 1': corr_matrix.index[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j],
                    'Absolute': abs(corr_matrix.iloc[i, j])
                })
    
    if strong_corrs:
        strong_corrs_df = pd.DataFrame(strong_corrs).sort_values('Absolute', ascending=False)
        st.dataframe(
            strong_corrs_df.style.background_gradient(subset=['Correlation'], cmap='RdYlGn', vmin=-1, vmax=1),
            hide_index=True
        )
    else:
        st.info(f"No correlations found above threshold {threshold}")


def display_distribution_comparison(df: pd.DataFrame, num_cols: List[str]):
    """Compare distributions of multiple numerical features"""
    st.markdown("### 📊 Distribution Comparison")
    
    if len(num_cols) < 1:
        st.info("No numerical columns available")
        return
    
    # Select columns to compare
    selected_cols = st.multiselect(
        "Select features to compare",
        num_cols,
        default=num_cols[:min(4, len(num_cols))],
        max_selections=6
    )
    
    if not selected_cols:
        return
    
    # Normalization option
    normalize = st.checkbox("Normalize distributions (0-1 scale)", value=False)
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Overlapping distributions
    for col in selected_cols:
        data = df[col].dropna()
        if normalize:
            data = (data - data.min()) / (data.max() - data.min())
        axes[0].hist(data, alpha=0.5, label=col, bins=30)
    
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Overlapping Distributions")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plots
    box_data = []
    labels = []
    for col in selected_cols:
        data = df[col].dropna()
        if normalize:
            data = (data - data.min()) / (data.max() - data.min())
        box_data.append(data)
        labels.append(col)
    
    axes[1].boxplot(box_data, labels=labels, patch_artist=True)
    axes[1].set_ylabel("Value")
    axes[1].set_title("Box Plot Comparison")
    axes[1].grid(alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_outlier_detection_report(df: pd.DataFrame, num_cols: List[str]):
    """Comprehensive outlier detection using multiple methods"""
    st.markdown("### 🎯 Outlier Detection Report")
    
    if not num_cols:
        st.info("No numerical columns available for outlier detection")
        return
    
    selected_col = st.selectbox("Select column for outlier analysis", num_cols)
    
    data = df[selected_col].dropna()
    
    # Multiple outlier detection methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### IQR Method")
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        st.metric("Outliers Found", len(iqr_outliers))
        st.write(f"Lower Bound: {lower_bound:.2f}")
        st.write(f"Upper Bound: {upper_bound:.2f}")
    
    with col2:
        st.markdown("#### Z-Score Method")
        z_scores = np.abs(stats.zscore(data))
        z_threshold = 3
        z_outliers = data[z_scores > z_threshold]
        st.metric("Outliers Found (Z > 3)", len(z_outliers))
        st.write(f"Mean: {data.mean():.2f}")
        st.write(f"Std Dev: {data.std():.2f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Box plot
    axes[0].boxplot(data, vert=True)
    axes[0].axhline(y=lower_bound, color='r', linestyle='--', label='IQR Bounds')
    axes[0].axhline(y=upper_bound, color='r', linestyle='--')
    axes[0].set_title(f"Box Plot - {selected_col}")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Histogram with outliers
    axes[1].hist(data, bins=50, alpha=0.7, color='blue', label='Data')
    if len(iqr_outliers) > 0:
        axes[1].hist(iqr_outliers, bins=20, alpha=0.7, color='red', label='IQR Outliers')
    axes[1].axvline(x=lower_bound, color='r', linestyle='--', label='IQR Bounds')
    axes[1].axvline(x=upper_bound, color='r', linestyle='--')
    axes[1].set_title(f"Distribution with Outliers - {selected_col}")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def display_feature_importance_analysis(df: pd.DataFrame, target_col: str, feature_cols: List[str]):
    """Analyze feature importance for a target variable"""
    st.markdown("### 🎯 Feature Importance Analysis")
    
    if not feature_cols or not target_col:
        st.info("Select features and target for importance analysis")
        return
    
    # Calculate correlations with target
    correlations = []
    for col in feature_cols:
        if col != target_col and pd.api.types.is_numeric_dtype(df[col]):
            corr = df[col].corr(df[target_col])
            correlations.append({
                'Feature': col,
                'Correlation': corr,
                'Absolute': abs(corr)
            })
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('Absolute', ascending=False)
        
        # Visualization
        fig = px.bar(
            corr_df,
            x='Feature',
            y='Correlation',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            title=f'Feature Correlations with {target_col}'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(corr_df, hide_index=True)


def display_time_series_analysis(df: pd.DataFrame, date_col: str, value_cols: List[str]):
    """Time series analysis and visualization"""
    st.markdown("### 📈 Time Series Analysis")
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            st.error(f"Cannot convert {date_col} to datetime")
            return
    
    selected_col = st.selectbox("Select value column", value_cols)
    
    # Sort by date
    df_sorted = df.sort_values(date_col)
    
    # Create interactive plot
    fig = px.line(
        df_sorted,
        x=date_col,
        y=selected_col,
        title=f"{selected_col} over Time"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Start Date", df_sorted[date_col].min().strftime('%Y-%m-%d'))
    with col2:
        st.metric("End Date", df_sorted[date_col].max().strftime('%Y-%m-%d'))
    with col3:
        date_range = (df_sorted[date_col].max() - df_sorted[date_col].min()).days
        st.metric("Date Range (days)", date_range)


def display_categorical_insights(df: pd.DataFrame, cat_cols: List[str]):
    """Deep dive into categorical variables"""
    st.markdown("### 🏷️ Categorical Variable Insights")
    
    if not cat_cols:
        st.info("No categorical columns found")
        return
    
    selected_col = st.selectbox("Select categorical column", cat_cols)
    
    # Value counts
    value_counts = df[selected_col].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Statistics")
        st.metric("Unique Values", df[selected_col].nunique())
        st.metric("Most Common", value_counts.index[0])
        st.metric("Most Common Count", value_counts.values[0])
        st.metric("Missing Values", df[selected_col].isnull().sum())
    
    with col2:
        # Pie chart
        fig = px.pie(
            values=value_counts.values[:10],
            names=value_counts.index[:10],
            title=f"Top 10 Values Distribution - {selected_col}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Frequency table
    st.markdown("#### Frequency Table")
    freq_df = pd.DataFrame({
        'Value': value_counts.index,
        'Count': value_counts.values,
        'Percentage': (value_counts.values / len(df) * 100).round(2)
    })
    st.dataframe(freq_df, hide_index=True)
