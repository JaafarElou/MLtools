import streamlit as st
import pandas as pd
from src.data_import import data_import_page
from src.data_visualization import data_visualization_page
from src.data_preparation import data_preparation_page
from src.ml_modeling import ml_modeling_page
import streamlit as st
from pathlib import Path

def home_page():
    """
    Enhanced Home Page for the ML Exploration App
    """
    # Header Title with Icon
    st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-family: 'Helvetica Neue', sans-serif; font-size: 3rem; color: #2C3E50;">
                🤖 <span style="color: #007BFF;">ML Exploration App</span>
            </h1>
            <p style="font-size: 1.2rem; color: #6C757D;">
                Votre guide interactif pour un workflow complet en data science.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Animated Text Section
    st.markdown("""
        <div style="margin: 30px auto; text-align: center;">
            <div class="animated-text">
                Bienvenue dans une plateforme où l'apprentissage machine devient facile !
                Découvrez toutes les étapes pour explorer et modéliser vos données.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
        <div style="display: flex; justify-content: center; gap: 30px; margin: 40px 0;">
            <div style="text-align: center; width: 20%; font-family: Arial, sans-serif;">
                <h3>📂</h3>
                <p><b>Importation</b><br>Données simplifiées.</p>
            </div>
            <div style="text-align: center; width: 20%; font-family: Arial, sans-serif;">
                <h3>📊</h3>
                <p><b>Visualisation</b><br>Exploration intuitive.</p>
            </div>
            <div style="text-align: center; width: 20%; font-family: Arial, sans-serif;">
                <h3>🧹</h3>
                <p><b>Préparation</b><br>Nettoyage optimisé.</p>
            </div>
            <div style="text-align: center; width: 20%; font-family: Arial, sans-serif;">
                <h3>🤖</h3>
                <p><b>Modélisation</b><br>Précision garantie.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Navigation Buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🗂️ Import Données"):
            st.session_state['current_page'] = 'import'
    with col2:
        if st.button("📊 Visualisation"):
            st.session_state['current_page'] = 'visualization'
    with col3:
        if st.button("🧹 Préparation"):
            st.session_state['current_page'] = 'preparation'
    with col4:
        if st.button("🤖 Modélisation"):
            st.session_state['current_page'] = 'modeling'

    # Tips Section
    with st.expander("💡 Conseils et Meilleures Pratiques"):
        st.markdown("""
        - Importez ou créez un dataset au préalable.
        - Visualisez vos données avant de les nettoyer.
        - Sélectionnez le modèle adéquat pour votre problème.
        - Évaluez et comparez vos résultats pour une meilleure compréhension.
        """)

def main_app():
    """
    Main Application Navigation
    """
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
    
    st.sidebar.title("🤖 ML Exploration")
    page = st.sidebar.radio(
        "Navigation",
        ["Accueil", "Import Données", "Visualisation", "Préparation", "Modélisation"]
    )
    if page == "Accueil":
        home_page()
    elif page == "Import Données":
        data_import_page()
    elif page == "Visualisation":
        data_visualization_page()
    elif page == "Préparation":
        data_preparation_page()
    elif page == "Modélisation":
        ml_modeling_page()

def app():
    """
    Application Entry Point
    """
    st.set_page_config(
        page_title="ML Exploration App",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Include Custom CSS
    st.markdown("""
        <style>
        /* General Styling */
        body { background-color: #F8F9FA; font-family: 'Arial', sans-serif; }
        .sidebar .sidebar-content { background: #FFFFFF; }
        .animated-text {
            font-size: 1.5rem;
            font-weight: 600;
            color: #007BFF;
            animation: slideIn 3s ease-out;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)
    main_app()

if __name__ == "__main__":
    app()