"""
app.py — Interfaz del Asistente de Optimización Administrativa
y Orientación Normativa UNEFA Lara
"""

import streamlit as st
from rag import responder

st.set_page_config(
    page_title="Asistente UNEFA Lara",
    page_icon="🎓",
    layout="centered",
)

st.title("🎓 Asistente de Orientación Normativa")
st.caption("UNEFA Núcleo Lara — Optimización Administrativa")

st.markdown(
    "Consulta reglamentos, normativas y procedimientos administrativos "
    "institucionales. Las respuestas se basan únicamente en los documentos "
    "oficiales cargados en el sistema."
)

with st.sidebar:
    st.header("Acerca de este asistente")
    st.write(
        "Este asistente responde con base en los documentos normativos "
        "y administrativos oficiales de UNEFA Lara cargados en su índice. "
        "No sustituye la consulta directa con las unidades administrativas "
        "correspondientes para trámites formales."
    )
    st.divider()
    if st.button("🗑️ Borrar historial de conversación"):
        st.session_state.historial = []
        st.rerun()

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

pregunta = st.chat_input(
    "Escribe tu pregunta sobre normativa o trámites administrativos..."
)

if pregunta:
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando los documentos normativos..."):
            try:
                respuesta, fuentes = responder(pregunta)
            except Exception as e:
                respuesta = (
                    "Ocurrió un error al consultar el asistente. "
                    f"Detalle técnico: {e}"
                )
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
