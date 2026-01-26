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
    data = joblib.load(
        "../data/modelo_precios_carburantes.pkl",
        mmap_mode="r"
    )
    return data["model"], data["columns"]

model, columns = load_model()



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
    "Badajoz", "Barcelona", "Burgos", "Cáceres", "Cádiz", "Cantabria",
    "Castellón/Castelló", "Ciudad Real", "Córdoba", "Cuenca", "Girona",
    "Granada", "Guadalajara", "Gipuzkoa", "Huelva", "Huesca",
    "Illes Balears", "Jaén", "La Rioja", "Las Palmas", "León", "Lleida",
    "Lugo", "Madrid", "Málaga", "Murcia", "Navarra", "Ourense", "Palencia",
    "Pontevedra", "Salamanca", "Santa Cruz de Tenerife", "Segovia",
    "Sevilla", "Soria", "Tarragona", "Teruel", "Toledo",
    "Valencia/València", "Valladolid", "Bizkaia", "Zamora", "Zaragoza"
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
)

if fecha.year >= 2026:
    st.warning(
        "⚠️ **Aviso sobre la predicción**\n\n"
        "El modelo ha sido entrenado con datos históricos hasta **2025**. "
        "Para fechas a partir de **2026**, la predicción puede ser menos precisa "
        "debido a cambios económicos, fiscales o del mercado energético."
    )


# =========================
# PREPARACIÓN DE FEATURES
# =========================
def preparar_datos(provincia, producto, fecha):
    df = pd.DataFrame({
        "provincia": [provincia],
        "producto": [producto],
        "pai": ["España"],
        "year": [fecha.year],
        "month": [fecha.month],
        "day": [fecha.day]
    })

    df = pd.get_dummies(df, drop_first=True)

    df = df.reindex(columns=columns, fill_value=0)

    return df


# =========================
# PREDICCIÓN
# =========================
st.divider()

if st.button("Predecir precio", use_container_width=True):
    X = preparar_datos(provincia, producto, fecha)

    precio = model.predict(X)[0]

    st.success("Predicción realizada correctamente")

    st.markdown(
    f"""
    <div style="
        background-color:#eaf2fb;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border:1px solid #c9ddf2;
        color:#0b3c5d;
        ">
        <h3 style="margin-bottom:10px;">Precio estimado</h3>
        <h1 style="color:#1f7a8c; margin:10px 0;">
            {precio:.3f} €/L
        </h1>
        <p style="color:#1c1c1c;">
            {producto}<br>
            {provincia}<br>
            {fecha.strftime('%d/%m/%Y')}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# PIE DE PÁGINA
# =========================
st.divider()
st.caption("Proyecto IA · Datos CNMC · Predicción orientativa")
