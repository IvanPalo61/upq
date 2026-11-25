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

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input(
    "Escribe tu pregunta..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role":"user",
            "content":prompt
        }
    )

    resultado = responder(prompt)

    respuesta = resultado["respuesta"]

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content":respuesta
        }
    )

    st.rerun()