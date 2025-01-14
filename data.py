import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def load_data():
    """Handles file upload and displays the data if available."""
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.success("Dataset uploaded successfully!")

        st.markdown("### Data Preview")
        st.dataframe(df.head())

        return df
    else:
        st.warning("Please upload a dataset to proceed.")
        return None


def display_summary(df):
    """Displays summary statistics and data insights."""
    st.sidebar.header("Summary Statistics")

    st.markdown("### Basic Information")
    st.write(f"**Number of Rows:** {df.shape[0]}")
    st.write(f"**Number of Columns:** {df.shape[1]}")
    st.write(f"**Missing Values:** {df.isnull().sum().sum()}")

    st.markdown("### Column Statistics")
    st.dataframe(df.describe().transpose())


def display_kpis(df):
    """Displays key performance indicators in card format."""
    st.markdown("### Key Performance Indicators")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())


def interactive_visualizations(df):
    """Provides interactive plots for data visualization."""
    st.sidebar.header("Visualization Settings")

    chart_type = st.sidebar.selectbox("Select Chart Type", ["Histogram", "Scatter Plot", "Box Plot"])
    columns = st.sidebar.multiselect("Select Columns", options=df.columns)

    if not columns:
        st.warning("Please select at least one column.")
        return

    if chart_type == "Histogram":
        for col in columns:
            fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        if len(columns) < 2:
            st.warning("Please select at least two columns for a scatter plot.")
            return
        x_col, y_col = columns[:2]
        fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        for col in columns:
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="Dynamic Dashboard", layout="wide")
    st.title("Dynamic Data Dashboard")

    df = load_data()

    if df is not None:
        display_summary(df)
        display_kpis(df)
        interactive_visualizations(df)


if __name__ == "__main__":
    main()
