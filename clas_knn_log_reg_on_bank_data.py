
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Prédiction bancaire",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Prédiction de souscription bancaire")
st.write(
    "Cette application utilise un modèle XGBoost "
    "pour prédire si un client va souscrire à un dépôt bancaire."
)

# ==================================================
# 2. CHARGEMENT DES FICHIERS
# ==================================================

@st.cache_resource
def load_models():
    model = joblib.load("xgb.joblib")
    scaler = joblib.load("scaler.joblib")
    encoders = joblib.load("encoders.joblib")
    uniques = joblib.load("uniques.joblib")
    return model, scaler, encoders, uniques


try:
    model, scaler, encoders, uniques = load_models()
except Exception as e:
    st.error(
        "Impossible de charger les fichiers du modèle. "
        "Vérifie que xgb.joblib, scaler.joblib, "
        "encoders.joblib et uniques.joblib sont présents."
    )
    st.exception(e)
    st.stop()


# ==================================================
# 3. VARIABLES DU MODÈLE
# ==================================================

# L'ordre correspond aux colonnes utilisées pour entraîner X.
FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]

# Indices des encodeurs pour les variables catégorielles.
ENCODER_INDEX = {
    "job": 0,
    "marital": 1,
    "education": 2,
    "default": 3,
    "housing": 4,
    "contact": 5,
    "month": 6,
    "day_of_week": 7,
    "poutcome": 8
}

# ==================================================
# 4. FONCTION DE PRÉPARATION DES DONNÉES
# ==================================================

def prepare_data(data):
    data = data.copy()

    # Vérifier que toutes les colonnes sont présentes.
    missing = [col for col in FEATURES if col not in data.columns]

    if missing:
        raise ValueError(
            "Colonnes manquantes dans les données : "
            + ", ".join(missing)
        )

    data = data[FEATURES].copy()

    # Encoder les variables catégorielles.
    for col, index in ENCODER_INDEX.items():
        encoder = encoders[index]

        values = data[col].astype(str).str.strip()

        unknown_values = set(values.unique()) - set(encoder.classes_)

        if unknown_values:
            raise ValueError(
                f"Valeur(s) inconnue(s) pour {col} : "
                f"{', '.join(sorted(unknown_values))}. "
                f"Valeurs autorisées : "
                f"{', '.join(map(str, encoder.classes_))}"
            )

        data[col] = encoder.transform(values)

    # Convertir les variables numériques.
    numeric_cols = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="raise")

    # Respecter l'ordre exact des colonnes du modèle.
    data = data[FEATURES]

    # Appliquer le scaler déjà entraîné.
    transformed = scaler.transform(data)

    return transformed


# ==================================================
# 5. FONCTION DE PRÉDICTION
# ==================================================

def predict_data(data):
    X = prepare_data(data)

    predictions = model.predict(X)

    # L'encodeur numéro 9 correspond à la cible y.
    target_encoder = encoders[9]
    predictions = target_encoder.inverse_transform(
        np.asarray(predictions).astype(int)
    )

    return predictions


# ==================================================
# 6. INTERFACE DE L'APPLICATION
# ==================================================

tab1, tab2 = st.tabs([
    "👤 Prédiction individuelle",
    "📁 Prédiction par fichier CSV"
])

# --------------------------------------------------
# ONGLET 1 : UN CLIENT
# --------------------------------------------------

with tab1:
    st.subheader("Informations du client")
    st.write("Renseigne les informations ci-dessous.")

    with st.form("client_form"):

        age = st.number_input(
            "Âge du client",
            min_value=18,
            max_value=100,
            value=35
        )

        job = st.selectbox(
            "Profession",
            options=list(encoders[0].classes_)
        )

        marital = st.selectbox(
            "Situation matrimoniale",
            options=list(encoders[1].classes_)
        )

        education = st.selectbox(
            "Niveau d'éducation",
            options=list(encoders[2].classes_)
        )

        default = st.selectbox(
            "Crédit en défaut ?",
            options=list(encoders[3].classes_)
        )

        housing = st.selectbox(
            "Prêt immobilier ?",
            options=list(encoders[4].classes_)
        )

        contact = st.selectbox(
            "Type de contact",
            options=list(encoders[5].classes_)
        )

        month = st.selectbox(
            "Mois du dernier contact",
            options=list(encoders[6].classes_)
        )

        day_of_week = st.selectbox(
            "Jour de la semaine",
            options=list(encoders[7].classes_)
        )

        duration = st.number_input(
            "Durée du dernier contact (secondes)",
            min_value=0,
            value=180
        )

        campaign = st.number_input(
            "Nombre de contacts pendant la campagne",
            min_value=1,
            value=1
        )

        pdays = st.number_input(
            "Nombre de jours depuis le contact précédent (999 = jamais contacté)",
            min_value=0,
            value=999
        )

        previous = st.number_input(
            "Nombre de contacts avant cette campagne",
            min_value=0,
            value=0
        )

        poutcome = st.selectbox(
            "Résultat de la campagne précédente",
            options=list(encoders[8].classes_)
        )

        submitted = st.form_submit_button(
            "🔍 Prédire",
            use_container_width=True
        )

    if submitted:
        client = pd.DataFrame([{
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }])

        try:
            result = predict_data(client)[0]

            if result == "yes":
                st.success(
                    "Résultat : le modèle prédit que le client "
                    "va souscrire au dépôt bancaire. (YES)"
                )
            else:
                st.info(
                    "Résultat : le modèle prédit que le client "
                    "ne va pas souscrire au dépôt bancaire. (NO)"
                )

            st.caption(
                "Il s'agit d'une prédiction statistique, "
                "pas d'une certitude."
            )

        except Exception as e:
            st.error(f"Erreur pendant la prédiction : {e}")


# --------------------------------------------------
# ONGLET 2 : PLUSIEURS CLIENTS
# --------------------------------------------------

with tab2:
    st.subheader("Prédiction pour plusieurs clients")

    st.write(
        "Importe un fichier CSV contenant les mêmes colonnes "
        "que celles utilisées lors de l'entraînement."
    )

    st.caption(
        "Le CSV doit contenir les 14 variables explicatives, "
        "avec leurs noms exacts en minuscules. "
        "La colonne cible y n'est pas nécessaire."
    )

    uploaded_file = st.file_uploader(
        "Choisir un fichier CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            csv_data = pd.read_csv(uploaded_file)

            st.write("Aperçu des données importées :")
            st.dataframe(csv_data.head())

            missing = [
                col for col in FEATURES
                if col not in csv_data.columns
            ]

            if missing:
                st.error(
                    "Colonnes manquantes : "
                    + ", ".join(missing)
                )
            elif st.button(
                "🚀 Prédire pour tous les clients",
                use_container_width=True
            ):
                results = predict_data(csv_data)

                output = csv_data.copy()
                output["prediction"] = results

                st.success(
                    f"Prédictions réalisées pour "
                    f"{len(output)} clients."
                )

                st.dataframe(output)

                csv_bytes = output.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="⬇️ Télécharger les résultats CSV",
                    data=csv_bytes,
                    file_name="predictions_bancaires.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:
            st.error(
                f"Impossible de traiter le fichier : {e}"
            )

st.divider()
st.caption("Application de prédiction — modèle XGBoost")




