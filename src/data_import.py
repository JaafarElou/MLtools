import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns
import requests  # To fetch data from URLs
from pathlib import Path


def data_import_page():
    """
    Page for importing and creating datasets with session persistence
    """
    st.title("🗂️ Import and Creation of Data")

    # Reset dataset button in the sidebar
    if st.sidebar.button("Reset Dataset"):
        for key in ['uploaded_data', 'data_source']:
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()

    # If a dataset is already loaded, display it with statistics
    if 'uploaded_data' in st.session_state:
        st.success("A dataset is already loaded. Explore it below.")
        show_dataset_and_statistics(st.session_state['uploaded_data'])
        return

    # Import method selection
    import_method = st.radio(
        "Select your import method",
        ["Local File", "Example Dataset", "URL (Web Link)", "Manual Creation"]
    )

    if import_method == "Local File":
        local_file_import()
    elif import_method == "Example Dataset":
        example_dataset_import()
    elif import_method == "URL (Web Link)":
        url_data_import()
    else:
        manual_data_creation()


def show_dataset_and_statistics(df):
    """
    Display the dataset and descriptive statistics
    """
    st.write("### Dataset")
    st.dataframe(df, height=300)

    st.write("### 📊 Descriptive Statistics")
    stats = df.describe().transpose()  # Transpose for better readability
    st.dataframe(stats, height=300)


def local_file_import():
    """
    Import files from the local computer
    """
    uploaded_file = st.file_uploader(
        "Upload your file",
        type=['csv', 'xlsx', 'json']
    )

    if uploaded_file is not None:
        try:
            # Load the file based on its type
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)

            # Save to session state
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'local file'
            st.success("File successfully imported!")
            show_dataset_and_statistics(df)
        except Exception as e:
            st.error(f"Error during import: {e}")


def url_data_import():
    """
    Import data from a URL
    """
    url = st.text_input("Enter the link to the file (CSV or JSON format)")

    if st.button("Load from URL"):
        if url:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check if the URL is valid

                # Detect file type based on extension
                if url.endswith('.csv'):
                    df = pd.read_csv(pd.compat.StringIO(response.text))
                elif url.endswith('.json'):
                    df = pd.read_json(pd.compat.StringIO(response.text))
                else:
                    st.error("Unsupported format. Only CSV and JSON files are accepted.")
                    return

                # Save to session state
                st.session_state['uploaded_data'] = df
                st.session_state['data_source'] = f'URL: {url}'
                st.success("Data successfully imported!")
                show_dataset_and_statistics(df)
            except Exception as e:
                st.error(f"Error during import: {e}")
        else:
            st.warning("Please enter a valid URL.")


def example_dataset_import():
    """
    Import example datasets
    """
    dataset_choices = {
        "Iris": load_iris(as_frame=True).frame,
        "Titanic": sns.load_dataset('titanic')
    }

    selected_dataset = st.selectbox(
        "Choose an example dataset",
        list(dataset_choices.keys())
    )

    if st.button("Load Dataset"):
        df = dataset_choices[selected_dataset]

        # Save to session state
        st.session_state['uploaded_data'] = df
        st.session_state['data_source'] = f'example dataset {selected_dataset}'

        st.success(f"Dataset {selected_dataset} loaded successfully!")
        show_dataset_and_statistics(df)


def manual_data_creation():
    """
    Manually create a dataset
    """
    st.write("Create your own dataset")

    # Number of rows
    num_rows = st.number_input("Number of rows", min_value=1, max_value=100, value=10)

    # Columns
    num_columns = st.number_input("Number of columns", min_value=1, max_value=10, value=3)

    columns = []
    data_types = {}

    for i in range(num_columns):
        col_name = st.text_input(f"Column name {i+1}")
        col_type = st.selectbox(f"Data type for {col_name}", ['Numeric', 'Categorical'])

        if col_name:
            columns.append(col_name)
            data_types[col_name] = col_type

    if st.button("Generate Dataset"):
        if len(columns) == num_columns:
            # Generate data
            df_data = {}
            for col, col_type in data_types.items():
                if col_type == 'Numeric':
                    df_data[col] = np.random.rand(num_rows) * 100
                else:
                    df_data[col] = np.random.choice(['A', 'B', 'C'], num_rows)

            df = pd.DataFrame(df_data)

            # Save to session state
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'manual creation'

            st.success("Dataset successfully created!")
            show_dataset_and_statistics(df)
        else:
            st.warning("Please name all columns")


if __name__ == "__main__":
    data_import_page()
