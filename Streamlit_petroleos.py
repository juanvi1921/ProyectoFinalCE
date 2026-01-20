import streamlit as st
import pandas as pd
import joblib
from datetime import date

# =========================
# CONFIGURACIÓN BÁSICA
# =========================
st.set_page_config(
    page_title="Predicción carburantes",
    page_icon="⛽",
    layout="centered"
)

# =========================
# CARGA DEL MODELO
# =========================
@st.cache_resource
def load_model():
    return joblib.load("../data/modelo_precios_carburantes.pkl")

model = load_model()

# =========================
# INTERFAZ
# =========================
st.title("⛽ Predicción de precios de carburantes")
st.write(
    "Introduce la **provincia**, el **tipo de carburante** y la **fecha** "
    "para obtener una estimación del precio."
)

st.divider()

# =========================
# LISTA DE PROVINCIAS DE ESPAÑA
# =========================
provincias = [
    "Álava", "Albacete", "Alicante/Alacant", "Almería", "Asturias", "Ávila",
    "Badajoz", "Barcelona", "Burgos", "Cáceres", "Cádiz", "Cantabria", "Castellón/Castelló",
    "Ciudad Real", "Córdoba", "Cuenca", "Girona", "Granada", "Guadalajara",
    "Gipuzkoa", "Huelva", "Huesca", "Illes Balears", "Jaén", "La Rioja",
    "Las Palmas", "León", "Lleida", "Lugo", "Madrid", "Málaga",
    "Murcia", "Navarra", "Ourense", "Palencia", "Pontevedra", "Salamanca",
    "Santa Cruz de Tenerife", "Segovia", "Sevilla", "Soria", "Tarragona",
    "Teruel", "Toledo", "Valencia/València", "Valladolid", "Bizkaia", "Zamora", "Zaragoza"
]

# =========================
# ENTRADAS DEL USUARIO
# =========================
provincia = st.selectbox("Provincia", provincias)

producto = st.selectbox(
    "Tipo de carburante",
    [
        "Gasolina 95 E5",
        "Gasolina 98 E5",
        "Gasóleo A habitual",
        "Gasóleo Premium"
    ]
)

fecha = st.date_input(
    "Fecha",
    min_value=date(2016, 1, 1),
    max_value=date(2025, 12, 31)
)

# =========================
# PREPARACIÓN DE FEATURES
# =========================
def preparar_datos(provincia, producto, fecha):
    return pd.DataFrame({
        "provincia": [provincia],
        "producto": [producto],
        "year": [fecha.year],
        "month": [fecha.month],
        "day": [fecha.day]
    })

# =========================
# PREDICCIÓN
# =========================
st.divider()

if st.button("🔮 Predecir precio", use_container_width=True):
    X = preparar_datos(provincia, producto, fecha)
    
    try:
        precio = model.predict(X)[0]
        st.success("Predicción realizada correctamente")

        st.markdown(
            f"""
            <div style="
                background-color:#f2f2f2;
                padding:20px;
                border-radius:10px;
                text-align:center;">
                <h3>Precio estimado</h3>
                <h1 style="color:#1f77b4;">{precio:.3f} €/L</h1>
                <p>
                {producto}<br>
                {provincia}<br>
                {fecha.strftime('%d/%m/%Y')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    except ValueError as e:
        st.error(f"Error en la predicción: {e}\n\n"
                 "Asegúrate de que el modelo fue entrenado correctamente con un pipeline que incluya OneHotEncoder para las variables categóricas.")

# =========================
# PIE DE PÁGINA
# =========================
st.divider()
st.caption("Proyecto IA · Datos CNMC · Predicción orientativa")
