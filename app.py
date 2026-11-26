import streamlit as st
from motor_chatbot import responder

st.set_page_config(
    page_title="Chatbot UPQ",
    page_icon="🎓"
)

st.title("🎓 Chatbot Oficial UPQ")

st.caption(
    "Asistente virtual para información oficial de la Universidad Politécnica de Querétaro"
)

# Botón para limpiar conversación
if st.button(" Limpiar conversación"):
    st.session_state.messages = []
    st.rerun()

# Historial
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if "extra" in msg:
            st.caption(msg["extra"])

# Entrada de usuario
prompt = st.chat_input("Escribe tu pregunta...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    resultado = responder(prompt)

    respuesta = resultado["respuesta"]

    if resultado["fallback"]:
        extra = (
            f" Fallback activado | "
            f"Confianza: {resultado['confianza']:.2f}"
        )
    else:
        extra = (
            f" Intención: {resultado['intencion']} | "
            f"Confianza: {resultado['confianza']:.2f}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta,
            "extra": extra
        }
    )

    st.rerun()