import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------- Configuration de la page ----------
st.set_page_config(page_title="Car Price App", page_icon="🚗", layout="centered")

# ---------- Style CSS personnalisé ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1E3D59, #3A6EA5, #6C63FF);
        background-attachment: fixed;
        color: #ffffff;
    }
    h1 {
        color: #FFD700;
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }
    h3 {
        color: #E0E0E0;
        text-align: center;
    }
    label, .stSelectbox label, .stNumberInput label {
        color: white !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color:#FFD700;
        color:#1E3D59;
        border-radius:12px;
        padding:10px 20px;
        font-weight:bold;
        border:none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color:#FFA500;
        color:white;
    }
    .result-box {
        background-color: rgba(255,255,255,0.1);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #FFD700;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #FFD700;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .form-container {
        background-color: rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    .car-card {
        background-color: rgba(255,255,255,0.12);
        border: 1px solid #FFD700;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0;
        color: #FFD700;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.25);
    }
    .car-card .title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 6px;
        color: #FFD700;
    }
    .car-card .meta {
        color: #F1F1F1;
        font-size: 14px;
    }
    .cards-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 12px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚗 Estimation du prix d'une voiture d'occasion")
st.markdown("### Modèle sélectionné : **XGBoost** (meilleur RMSE prix)")

# ---------- Chargement du modèle ----------
MODEL_PATH = "best_model_xgb.pkl"
COLUMNS_PATH = "model_columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers: {e}")
    st.stop()

# ---------- Mappings UI -> tokens d'entraînement ----------
UI_TO_TRAINING = {
    "brand": {
        "renault": "Renault","peugeot": "Peugeot","volkswagen": "Volkswagen","toyota": "Toyota","kia": "Kia",
        "hyundai": "Hyundai","bmw": "BMW","mercedes-benz": "Mercedes-Benz","audi": "Audi","porsche": "Porsche",
        "volvo": "Volvo","citroen": "Citroen","dacia": "Dacia","nissan": "Nissan","opel": "Opel","seat": "Seat",
        "skoda": "Skoda","suzuki": "Suzuki","honda": "Honda","mazda": "Mazda","mitsubishi": "Mitsubishi",
        "lexus": "Lexus","land rover": "Land Rover","jeep": "Jeep","wallyscar": "wallyscar","autres": "Autres",
        "inconnu": "unknown"
    },
    "fuel": {"essence":"Essence","diesel":"Diesel","electrique":"Electrique"},
    "gearbox": {"manuelle":"Manuelle","automatique":"unknown"},
    "vehicle_condition": {"neuf":"Nouveau","très bon":"RS","bon":"Non dédouanné","moyen":"Pièces manquantes","inconnu":"unknown"},
    "location": {"tunis":"Tunis","sfax":"Sfax","sousse":"Sousse","nabeul":"Nabeul","gabès":"Gabès","bizerte":"Bizerte",
                 "ben arous":"Ben Arous","béja":"Béja","gafsa":"Gafsa","jendouba":"Jendouba","kairouan":"Kairouan",
                 "kasserine":"Kasserine","kébili":"Kébili","la manouba":"La Manouba","le kef":"Le Kef","mahdia":"Mahdia",
                 "monastir":"Monastir","médenine":"Médenine","sidi bouzid":"Sidi Bouzid","siliana":"Siliana",
                 "tataouine":"Tataouine","tozeur":"Tozeur","zaghouan":"Zaghouan"}
}

def map_ui_value(field, value):
    return UI_TO_TRAINING.get(field, {}).get(value, value)

def prepare_features(user_inputs, model_columns):
    df = pd.DataFrame([user_inputs])
    numeric_cols = ["year","mileage","engine_power"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    cat_cols = ["brand","fuel","gearbox","vehicle_condition","location"]
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].apply(lambda v: map_ui_value(col, v))
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

# ---------- Interface utilisateur ----------
with st.form("car_form"):
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.subheader("Caractéristiques du véhicule")

    col1, col2, col3 = st.columns(3)

    with col1:
        year = st.number_input("Année", min_value=1990, max_value=2025, value=2015, step=1)
        mileage = st.number_input("Kilométrage (km)", min_value=0, max_value=1_000_000, value=120_000, step=1000)

    with col2:
        engine_power = st.number_input("Puissance (ch)", min_value=30, max_value=600, value=110, step=1)
        brand = st.selectbox("Marque", list(UI_TO_TRAINING["brand"].keys()), index=0)

    with col3:
        fuel = st.selectbox("Carburant", list(UI_TO_TRAINING["fuel"].keys()), index=0)
        gearbox = st.selectbox("Boîte de vitesses", list(UI_TO_TRAINING["gearbox"].keys()), index=0)
        vehicle_condition = st.selectbox("État du véhicule", list(UI_TO_TRAINING["vehicle_condition"].keys()), index=0)
        location = st.selectbox("Localisation", list(UI_TO_TRAINING["location"].keys()), index=0)

    submitted = st.form_submit_button("🔍 Prédire le prix")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Prédiction ----------
if submitted:
    user_inputs_ui = {
        "year": year,
        "mileage": mileage,
        "engine_power": engine_power,
        "brand": brand,
        "fuel": fuel,
        "gearbox": gearbox,
        "vehicle_condition": vehicle_condition,
        "location": location,
    }
    try:
        X = prepare_features(user_inputs_ui, model_columns)
        y_pred_log = model.predict(X)
        price_pred = np.expm1(y_pred_log)
        prix_estime = float(price_pred[0])

        # ---------- Ajustement manuel basé sur la puissance ----------
        if engine_power > 100:
            prix_estime *= (1 + (engine_power - 100) * 0.005)

        st.markdown(f"<div class='result-box'>💰 Prix estimé : {prix_estime:,.0f} TND</div>", unsafe_allow_html=True)

        # ---------- Exemples de voitures proches ----------
        st.subheader("🚘 Exemples de voitures avec prix similaire")

        # Dataset fictif d'exemples (tu peux remplacer par des données réelles)
        exemples_voitures = [
            {"marque": "Renault Clio 2018", "prix": 39500, "details": "Essence · Manuelle · 110 ch"},
            {"marque": "Peugeot 208 2019", "prix": 41000, "details": "Diesel · Manuelle · 100 ch"},
            {"marque": "Volkswagen Polo 2017", "prix": 42000, "details": "Essence · Automatique · 95 ch"},
            {"marque": "Hyundai i20 2019", "prix": 38500, "details": "Essence · Manuelle · 100 ch"},
            {"marque": "Toyota Yaris 2018", "prix": 40500, "details": "Essence · Manuelle · 100 ch"},
            {"marque": "Kia Rio 2018", "prix": 39000, "details": "Essence · Manuelle · 100 ch"},
            {"marque": "Skoda Fabia 2017", "prix": 37000, "details": "Diesel · Manuelle · 90 ch"},
            {"marque": "Citroen C3 2019", "prix": 40000, "details": "Essence · Automatique · 110 ch"},
        ]

        # Filtrer les voitures proches du prix estimé (±10%)
        marge = prix_estime * 0.10
        voitures_proches = [v for v in exemples_voitures if abs(v["prix"] - prix_estime) <= marge]

        if voitures_proches:
            st.markdown("<div class='cards-grid'>", unsafe_allow_html=True)
            for v in voitures_proches:
                st.markdown(
                    f"""
                    <div class='car-card'>
                        <div class='title'>{v["marque"]}</div>
                        <div class='meta'>Prix moyen: {v["prix"]:,.0f} TND</div>
                        <div class='meta'>{v["details"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Aucune voiture exemple trouvée dans cette gamme de prix.")

    except Exception as e:
        st.error(f"Erreur pendant la prédiction: {e}")

# ---------- Footer ----------
st.markdown("---")
st.caption("✨ Application développée avec Streamlit — Projet CarPriceApp")
