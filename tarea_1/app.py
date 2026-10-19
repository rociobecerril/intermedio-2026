"""Dashboard Streamlit – Tarea 1: Noticias peruanas, del titular al clasificador."""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).parent
MODELO_DIR = BASE / "modelo"
CATS = ["politica", "economia", "deportes", "ciencia_tecnologia"]

st.set_page_config(page_title="Noticias peruanas: del titular al clasificador", page_icon="🗞️", layout="wide")


@st.cache_data
def cargar_datos():
    return pd.read_csv(BASE / "datos" / "noticias.csv")


@st.cache_data
def cargar_metricas():
    ruta = MODELO_DIR / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


@st.cache_resource
def cargar_modelo():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODELO_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODELO_DIR).eval()
    return tok, model, torch


def clasificar(titular: str):
    tok, model, torch = cargar_modelo()
    enc = tok(titular, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0].tolist()
    return {model.config.id2label[i]: p for i, p in enumerate(probs)}


df = cargar_datos()
met = cargar_metricas()

# ---------- Encabezado ----------
st.markdown("<h1 style='text-align:center'>🗞️ ¿Pueden convertir titulares peruanos en un clasificador de noticias?</h1>",
            unsafe_allow_html=True)
st.markdown(
    """
**Resumen metodológico.** Se extrajeron titulares reales de **La República** (RSS, sitemaps y notas individuales),
se normalizaron a 4 categorías (`politica`, `economia`, `deportes`, `ciencia_tecnologia`) y se etiquetaron con un **LLM**
(Gemini) para compararlo con la etiqueta de sección y con una **auditoría humana** de 20 casos. Luego se afinó un
**DistilBERT multilingüe** con split estratificado 70/15/15 y se evaluó solo en el conjunto de **test** reservado.
Abajo puedes explorar los datos y probar el clasificador en vivo.
"""
)

# ---------- Sidebar ----------
st.sidebar.header("Filtros")
medios = sorted(df["medio"].unique())
sel_medios = st.sidebar.multiselect("Medio", medios, default=medios)
sel_cats = st.sidebar.multiselect("Categoría", CATS, default=CATS)
filtrado = df[df["medio"].isin(sel_medios) & df["etiqueta"].isin(sel_cats)]
st.sidebar.caption(f"{len(filtrado)} de {len(df)} noticias con los filtros actuales")

# ---------- Indicadores ----------
acuerdo = met.get("acuerdo_llm_humano")
c1, c2, c3 = st.columns(3)
c1.metric("Noticias recolectadas", f"{len(df)}", delta=f"{len(filtrado)} en el filtro", delta_color="off")
c2.metric("F1 macro en test", f"{met['f1_macro_test']:.2f}" if "f1_macro_test" in met else "—",
          help="Modelo DistilBERT fine-tuned evaluado sobre el conjunto de test reservado.")
c3.metric("Acuerdo LLM vs humano", f"{acuerdo}/20 ({acuerdo / 20:.0%})" if acuerdo is not None else "pendiente",
          help="Auditoría manual de 20 etiquetas al azar generadas por el LLM.")

# ---------- Visualización ----------
st.subheader("Distribución de categorías")
if filtrado.empty:
    st.info("Los filtros no dejan ninguna noticia. Amplía la selección en la barra lateral.")
else:
    conteo = filtrado["etiqueta"].value_counts().reindex(CATS).dropna().reset_index()
    conteo.columns = ["etiqueta", "n"]
    top = conteo.iloc[conteo["n"].argmax()]
    fig = px.bar(conteo, x="etiqueta", y="n", color="etiqueta", text="n",
                 title=f"{top['etiqueta']} lidera con {int(top['n'])} de {len(filtrado)} noticias filtradas",
                 labels={"etiqueta": "Categoría", "n": "Noticias"})
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Noticias")
    st.dataframe(filtrado[["titular", "bajada", "medio", "seccion_original", "fecha", "etiqueta", "url"]],
                 width="stretch", hide_index=True,
                 column_config={"url": st.column_config.LinkColumn("url")})
    st.download_button("⬇️ Descargar CSV filtrado", filtrado.to_csv(index=False).encode("utf-8"),
                       file_name="noticias_filtradas.csv", mime="text/csv")

# ---------- Probador en vivo ----------
st.divider()
st.subheader("🔎 Probador en vivo")
titular = st.text_input("Escribe o pega un titular", placeholder="Ej.: Congreso aprueba nuevo retiro de fondos de AFP")
if st.button("Clasificar", type="primary"):
    if not titular.strip():
        st.warning("Escribe un titular primero.")
    elif not (MODELO_DIR / "model.safetensors").exists():
        st.error("No se encontraron los pesos del modelo en `modelo/`. Ejecuta `02_clasificacion.ipynb` para entrenarlo.")
    else:
        with st.spinner("Clasificando..."):
            probs = clasificar(titular.strip())
        pred = max(probs, key=probs.get)
        st.success(f"Categoría predicha: **{pred}** — probabilidad {probs[pred]:.1%}")
        dfp = pd.DataFrame({"categoria": list(probs), "probabilidad": list(probs.values())}).sort_values("probabilidad")
        figp = px.bar(dfp, x="probabilidad", y="categoria", orientation="h", text=dfp["probabilidad"].map("{:.1%}".format),
                      range_x=[0, 1], title="Probabilidad por categoría")
        st.plotly_chart(figp, width="stretch")
