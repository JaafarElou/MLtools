import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt



def data_visualization_page():
    """
    Dashboard interactif centré pour la visualisation des données.
    """
    st.title("📊 Dashboard Interactif")

    # Vérifier si un dataset est chargé
    if 'uploaded_data' not in st.session_state:
        st.warning("Veuillez d'abord importer ou créer un dataset sur la page d'Import.")
        return

    df = st.session_state['uploaded_data']

    # Sidebar pour les options de visualisation
    st.sidebar.header("Options de Visualisation")
    viz_type = st.sidebar.radio(
        "Choisissez une Visualisation", 
        ["📈 Graphique de Tendances", "📊 Histogramme", "🔗 Matrice de Corrélation"]
    )

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    # Centre l'ensemble des éléments pour éviter un look dispersé
    col1, col2, col3 = st.columns([1, 5, 1])  # Centralize plots with spacing
    
    # Graphique de tendances (Line Chart)
    if viz_type == "📈 Graphique de Tendances":
        with col2:  # Center content
            st.subheader("Graphique de Tendances")
            x_axis = st.selectbox("Sélectionnez l'axe X", numeric_columns)
            y_axis = st.selectbox("Sélectionnez l'axe Y", numeric_columns)
            fig = px.line(df, x=x_axis, y=y_axis, title="Graphique de Tendances")
            st.plotly_chart(fig, use_container_width=True)

    # Histogramme interactif
    elif viz_type == "📊 Histogramme":
        with col2:  # Center content
            st.subheader("Histogramme des Variables")
            col_to_plot = st.selectbox("Sélectionnez une colonne", numeric_columns)
            fig = px.histogram(df, x=col_to_plot, marginal='box', title="Distribution de la Variable")
            st.plotly_chart(fig, use_container_width=True)

    # Matrice de corrélation
    elif viz_type == "🔗 Matrice de Corrélation":
        with col2:  # Center content
            st.subheader("Matrice de Corrélation")
            corr_matrix = df[numeric_columns].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            st.pyplot(plt)

if __name__ == "__main__":
    data_visualization_page()
