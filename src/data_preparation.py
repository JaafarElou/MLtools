import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreparationManager:
    def __init__(self, df):
        self.original_df = df.copy()
        self.prepared_df = df.copy()
        self.transformed = False  # Track if pre-treatment steps have been applied

    def handle_missing_values(self, strategy_dict):
        for col, strategy in strategy_dict.items():
            imputer = SimpleImputer(strategy=strategy)
            self.prepared_df[[col]] = imputer.fit_transform(self.prepared_df[[col]])
        self.transformed = True

    def normalize_data(self, method, columns):
        scaler = StandardScaler() if method == "StandardScaler" else MinMaxScaler()
        self.prepared_df[columns] = scaler.fit_transform(self.prepared_df[columns])
        self.transformed = True

    def encode_categorical(self, method, columns):
        if method == "Label Encoding":
            le = LabelEncoder()
            for col in columns:
                self.prepared_df[col] = le.fit_transform(self.prepared_df[col].astype(str))
        elif method == "One-Hot Encoding":
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(self.prepared_df[columns])
            encoded_cols = [f"{col}_{cat}" for col, cats in zip(columns, encoder.categories_) for cat in cats]
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=self.prepared_df.index)
            self.prepared_df = pd.concat([self.prepared_df.drop(columns=columns), encoded_df], axis=1)
        self.transformed = True

def data_preparation_page():
    st.title("🧹 Préparation des Données")

    if 'uploaded_data' not in st.session_state:
        st.warning("Veuillez d'abord importer ou créer un dataset sur la page d'Import.")
        return

    if 'data_prep_manager' not in st.session_state:
        st.session_state.data_prep_manager = DataPreparationManager(st.session_state['uploaded_data'])

    prep_manager = st.session_state.data_prep_manager

    # Step 1: Show Original Data
    st.subheader("📊 Données Originales")
    st.dataframe(prep_manager.original_df)

    # --- Gestion des Valeurs Manquantes ---
    st.subheader("🗐 Gestion des Valeurs Manquantes")
    missing_cols = prep_manager.prepared_df.columns[prep_manager.prepared_df.isnull().any()].tolist()

    if missing_cols:
        strategy_dict = {}
        for col in missing_cols:
            strategy = st.selectbox(f"Stratégie pour '{col}'", ["mean", "median", "most_frequent", "constant"],
                                    key=f"missing_{col}")
            strategy_dict[col] = strategy
        if st.button("Traiter les Valeurs Manquantes"):
            prep_manager.handle_missing_values(strategy_dict)
            st.success("Valeurs manquantes traitées avec succès!")
            st.experimental_rerun()
    else:
        st.info("Aucune valeur manquante détectée.")

    # --- Normalisation ---
    st.subheader("📊 Normalisation")
    numeric_cols = prep_manager.prepared_df.select_dtypes(include=['int64', 'float64']).columns
    normalization_method = st.selectbox("Méthode de Normalisation", ["Aucune", "StandardScaler", "MinMaxScaler"],
                                        key="norm_method")
    selected_norm_cols = st.multiselect("Colonnes à normaliser", numeric_cols, key="norm_cols")

    if st.button("Normaliser") and normalization_method != "Aucune":
        prep_manager.normalize_data(normalization_method, selected_norm_cols)
        st.success("Colonnes normalisées avec succès!")
        st.experimental_rerun()

    # --- Encodage des Variables Catégorielles ---
    st.subheader("🏷️ Encodage des Variables Catégorielles")
    cat_cols = prep_manager.prepared_df.select_dtypes(include=['object', 'category']).columns
    encoding_method = st.selectbox("Méthode d'Encodage", ["Aucun", "Label Encoding", "One-Hot Encoding"],
                                    key="encode_method")
    selected_encoding_cols = st.multiselect("Colonnes à encoder", cat_cols, key="encode_cols")

    if st.button("Encoder") and encoding_method != "Aucun":
        prep_manager.encode_categorical(encoding_method, selected_encoding_cols)
        st.success("Colonnes encodées avec succès!")
        st.experimental_rerun()

    # Step 2: Show Prepared Data **AFTER transformations**
    if prep_manager.transformed:
        st.subheader("🔄 Données Préparées")
        st.dataframe(prep_manager.prepared_df)

    # Save Prepared Data
    if st.button("🔍 Sauvegarder les Données Préparées"):
        st.session_state['uploaded_data'] = prep_manager.prepared_df
        st.success("Données préparées sauvegardées avec succès!")

if __name__ == "__main__":
    data_preparation_page()
