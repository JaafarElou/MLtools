import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    mean_squared_error, 
    r2_score, 
    accuracy_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor


class AdvancedModelTrainer:
    def __init__(self):
        """
        Initialize advanced model trainer with comprehensive model options
        """
        self.classification_models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "Neural Network": MLPClassifier()
        }

        self.regression_models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regression": SVR(),
            "Neural Network": MLPRegressor()
        }

    def preprocess_data(self, X, y):
        """
        Preprocess data for model training
        """
        # Encode categorical target variable
        le = LabelEncoder()
        y = le.fit_transform(y) if y.dtype == 'object' else y

        # Scale numeric features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X, y, le, scaler

    def validate_model(self, model, X, y, cv_folds=5):
        """
        Validate the model using cross-validation
        """
        scores = cross_val_score(model, X, y, cv=cv_folds)
        return scores

    def train_model(self, model, X_train, X_test, y_train, y_test, problem_type):
        """
        Train and evaluate model
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == 'classification':
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            return accuracy, report, conf_matrix

        else:  # regression
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return mse, r2, y_test, y_pred

def plot_confusion_matrix(conf_matrix, model_name):
    """
    Plot the confusion matrix
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.ylabel('Véritable')
    plt.xlabel('Prédit')
    st.pyplot(plt)

def plot_regression_results(y_test, y_pred):
    """
    Plot regression results
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot(y_test, y_test, color='red', linestyle='--')
    plt.title("Résultats de Régression")
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    st.pyplot(plt)

def ml_modeling_page():
    """
    Advanced Machine Learning Modeling Page
    """
    st.title("🤖 Machine Learning Model Training")

    # Check if data is loaded
    if 'uploaded_data' not in st.session_state:
        st.warning("Veuillez d'abord importer ou préparer un dataset.")
        return

    df = st.session_state['uploaded_data']

    # Sidebar for model configuration
    st.sidebar.header("🔧 Configuration du Modèle")

    # Problem Type Selection
    problem_type = st.sidebar.selectbox(
        "Type de Problème", 
        ["Classification", "Régression"]
    )

    # Feature and Target Selection
    st.sidebar.subheader("Sélection des Variables")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    feature_columns = st.sidebar.multiselect(
        "Colonnes de Features", 
        list(numeric_columns) + list(categorical_columns)
    )
    target_column = st.sidebar.selectbox(
        "Colonne Cible", 
        list(numeric_columns) + list(categorical_columns)
    )

    if not feature_columns or not target_column:
        st.warning("Veuillez sélectionner des features et une variable cible.")
        return

    X = df[feature_columns]
    y = df[target_column]

    # Model Selection
    trainer = AdvancedModelTrainer()
    selected_model_name = st.sidebar.selectbox(
        "Choisir un Modèle", 
        list(trainer.classification_models.keys()) if problem_type == "Classification" 
        else list(trainer.regression_models.keys())
    )
    selected_model = (
        trainer.classification_models[selected_model_name]
        if problem_type == "Classification" else trainer.regression_models[selected_model_name]
    )

    # Train-test split
    test_size = st.sidebar.slider("Taille de l'ensemble de test", 0.1, 0.3, 0.2)

    if st.sidebar.button("🚀 Entraîner le Modèle"):
        # Preprocess data
        X_processed, y_processed, label_encoder, scaler = trainer.preprocess_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=test_size, random_state=42)

        # Model validation
        with st.spinner("Validation du modèle en cours..."):
            validation_scores = trainer.validate_model(selected_model, X_processed, y_processed)
            st.sidebar.write(f"Validation croisée - Moyenne des Scores: {np.mean(validation_scores):.2%}")

        # Train model
        if problem_type == "Classification":
            accuracy, report, conf_matrix = trainer.train_model(selected_model, X_train, X_test, y_train, y_test, 'classification')
            st.subheader("🏆 Résultats du Modèle")
            st.metric("Précision", f"{accuracy:.2%}")
            st.subheader("📊 Rapport de Classification")
            st.dataframe(pd.DataFrame(report).transpose())
            plot_confusion_matrix(conf_matrix, selected_model_name)
        else:
            mse, r2, y_test, y_pred = trainer.train_model(selected_model, X_train, X_test, y_train, y_test, 'regression')
            st.subheader("🏆 Résultats du Modèle")
            st.metric("Erreur Quadratique Moyenne", f"{mse:.4f}")
            st.metric("R² Score", f"{r2:.2%}")
            plot_regression_results(y_test, y_pred)

if __name__ == "__main__":
    ml_modeling_page()
