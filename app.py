"""
app.py — Interfaz del Asistente de Optimización Administrativa
y Orientación Normativa UNEFA Lara
"""

import streamlit as st
from rag import responder

# 1. Definimos una función de carga con caché para que sea rápido
@st.cache_resource
def cargar_sistema():
    # Esto asegura que el sistema se inicialice una vez y se guarde en memoria
    return True

st.set_page_config(
    page_title="Asistente UNEFA Lara",
    page_icon="🎓",
    layout="centered",
)

# Inicializar sistema
cargar_sistema()

st.title("🎓 Asistente de Orientación Normativa")
st.caption("UNEFA Núcleo Lara — Optimización Administrativa")

# ... (resto de tu código de interfaz hasta el historial) ...

if "historial" not in st.session_state:
    st.session_state.historial = []

# Mostrar historial previo
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])
        if mensaje["rol"] == "assistant" and mensaje.get("fuentes"):
            with st.expander("📄 Fuentes consultadas"):
                for f in mensaje["fuentes"]:
                    st.markdown(
                        f"- **{f['fuente']}** — fragmento {f['chunk_id']} "
                        f"(relevancia: {f['similitud']:.2f})"
                    )

pregunta = st.chat_input("Escribe tu pregunta...")

if pregunta:
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando los documentos normativos..."):
            try:
                # La función responder ya está en rag.py
                respuesta, fuentes = responder(pregunta)
            except Exception as e:
                respuesta = f"Ocurrió un error al consultar el asistente. Detalle: {e}"
                fuentes = []

            st.markdown(respuesta)
            if fuentes:
                with st.expander("📄 Fuentes consultadas"):
                    for f in fuentes:
                        st.markdown(
                            f"- **{f['fuente']}** — fragmento {f['chunk_id']} "
                            f"(relevancia: {f['similitud']:.2f})"
                        )

    st.session_state.historial.append(
        {"rol": "assistant", "contenido": respuesta, "fuentes": fuentes}
    )
