# =========================================================
# Market Prices Data Product
# =========================================================

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests
from dotenv import load_dotenv

# =========================================================
# CONFIGURATION PAGE
# =========================================================

st.set_page_config(
    page_title="Market Prices Monitoring - West Africa",
    layout="wide"
)

st.title("Market Prices Monitoring in West Africa")
st.markdown(
    """
    **End-to-End Data Engineering & AI Project**  
    Monitoring food market prices using data pipelines, quality control,
    interactive dashboards and AI-powered economic insights.
    """
)

# =========================================================
# CONFIGURATION API GEMINI
# =========================================================

# In production: use st.secrets or environment variables
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY", "YOUR_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash")

# =========================================================
# FOLDERS
# =========================================================

folders = ["data/raw", "data/clean", "data/product", "logs", "docs"]
for f in folders:
    os.makedirs(f, exist_ok=True)

# =========================================================
# GENERATION DES DONNEES
# =========================================================

@st.cache_data
def generate_data():

    pays = ["Benin", "Ghana", "Togo"]
    villes = {
        "Benin": ["Cotonou", "Porto-Novo", "Parakou", "Abomey", "Bohicon"],
        "Ghana": ["Accra", "Kumasi", "Tamale", "Takoradi", "Cape Coast"],
        "Togo": ["Lome", "Sokode", "Kara", "Aneho", "Dapaong"]
    }

    produits = [
        "ble", "riz", "huile", "sucre", "wheat",
        "lait", "oeufs", "poisson", "haricots", "tomates"
    ]

    units = {
        "ble": "kg", "riz": "kg", "huile": "litre", "sucre": "kg",
        "wheat": "kg", "lait": "litre", "oeufs": "douzaine",
        "poisson": "kg", "haricots": "kg", "tomates": "kg"
    }

    currencies = {"Benin": "XOF", "Togo": "XOF", "Ghana": "GHS"}

    today = datetime.now()
    dates = [today - timedelta(days=i) for i in range(30)]

    rows = []
    for date in dates:
        for country in pays:
            for city in villes[country]:
                for product in produits:
                    rows.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "country": country.lower(),
                        "city": city.lower(),
                        "market": f"market_{np.random.randint(1, 6)}",
                        "product": product.lower(),
                        "price": round(np.random.uniform(100, 1000), 2),
                        "unit": units.get(product, "unit"),
                        "currency": currencies[country],
                        "source": "field survey",
                        "collected_by": f"agent_{np.random.randint(1, 10)}"
                    })

    df = pd.DataFrame(rows)
    df.to_csv("data/raw/market_prices_raw.csv", index=False)
    return df


# =========================================================
# NETTOYAGE
# =========================================================


def clean_data(df):
    df = df.drop_duplicates()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    df["price"] = df["price"].astype(float)

    for col in ["product", "unit", "country", "city", "market"]:
        df[col] = df[col].str.lower().str.strip()

    df.to_csv("data/clean/market_prices_clean.csv", index=False)
    return df


# =========================================================
# INDICATEURS QUALITE
# =========================================================


def compute_quality_metrics(df):
    total_rows = len(df)
    total_cells = df.shape[0] * df.shape[1]

    return {
        "nombre_lignes": total_rows,
        "taux_valeurs_manquantes": round(df.isnull().sum().sum() / total_cells, 4),
        "taux_doublons": round(df.duplicated().sum() / total_rows, 4),
        "prix_anormaux": round((df["price"] <= 0).sum() / total_rows, 4),
        "couverture_pays": df["country"].nunique(),
        "couverture_marches": df["market"].nunique(),
        "fraicheur_jours": (datetime.now() - df["date"].max()).days
    }


# =========================================================
# PIPELINE DATA
# =========================================================


df_raw = generate_data()
df_clean = clean_data(df_raw)
df_product = df_clean.copy()
df_product["ingestion_date"] = datetime.now()
df_product.to_csv("data/product/market_prices_daily.csv", index=False)

metrics = compute_quality_metrics(df_product)

# =========================================================
# SIDEBAR - FILTRES
# =========================================================

st.sidebar.header("Filters")

country_filter = st.sidebar.selectbox(
    "Country",
    [""] + sorted(df_product["country"].unique())
)

product_filter = st.sidebar.selectbox(
    "Product",
    [""] + sorted(df_product["product"].unique())
)


df_view = df_product.copy()
if country_filter:
    df_view = df_view[df_view["country"] == country_filter]
if product_filter:
    df_view = df_view[df_view["product"] == product_filter]

# =========================================================
# AFFICHAGE DONNEES
# =========================================================

st.subheader("Market Prices Data")
st.dataframe(df_view, use_container_width=True)

# =========================================================
# INDICATEURS QUALITE
# =========================================================

st.subheader("Data Quality Indicators")
st.dataframe(pd.DataFrame([metrics]))

# =========================================================
# ANALYSE IA
# =========================================================

st.subheader("AI-powered Economic Analysis")

if st.button("Generate AI Analysis"):
    with st.spinner("Analyzing prices..."):
        summary = df_product.groupby("product")["price"].mean().reset_index()

        prompt = f"""
        Tu es un expert en economie et data science.
        Analyse les prix moyens suivants et fournis :
        1. Resume des tendances
        2. Observation cle
        3. Recommandation economique

        Donnees :
        {summary.to_string(index=False)}
        """

        try:
            response = model.generate_content(prompt)
            st.success("Analysis generated")
            st.markdown(response.text)

            with open("docs/ai_analysis_report.txt", "w", encoding="utf-8") as f:
                f.write(response.text)

        except TooManyRequests:
            st.error("Quota API atteint. Reessayez plus tard.")
