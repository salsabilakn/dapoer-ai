# dapoer_ai_app.py
import streamlit as st
from dapoer_module import create_agent

st.set_page_config(page_title="🍲 Dapoer-AI", page_icon="🍛")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Input API Key
api_key = st.text_input("🔑 Masukkan API Key Gemini kamu:", type="password")
if not api_key:
    st.warning("Silakan masukkan API Key untuk mulai.")
    st.stop()

# Buat agent
agent = create_agent(api_key)

# Inisialisasi chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hai! Mau masak apa hari ini? Tanyakan judul, bahan, metode, atau minta rekomendasi masakan."
    })

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pertanyaan user
if prompt := st.chat_input("Tanyakan resep, bahan, metode, atau minta rekomendasi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"⚠️ Error: {e}"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
