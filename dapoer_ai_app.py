# dapoer_ai_final.py

import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Coba import Gemini (jika ada)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory
    USE_GEMINI = True
except ImportError:
    USE_GEMINI = False

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/audreeynr/dapoer-ai/main/data/Indonesian_Food_Recipes.csv"
    df = pd.read_csv(url)
    df = df.dropna(subset=["Title", "Ingredients", "Steps"]).drop_duplicates()
    df["Ingredients_Norm"] = df["Ingredients"].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    df["Steps_Norm"] = df["Steps"].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    return df

df = load_data()

def format_recipe(row):
    bahan = [f"- {b.strip().capitalize()}" for b in re.split(r'\n|,|--', row['Ingredients']) if b.strip()]
    return f"""🍽 **{row['Title']}**

**Bahan-bahan:**  
{chr(10).join(bahan)}

**Langkah-langkah:**  
{row['Steps'].strip()}"""

# Pencarian berdasarkan TF-IDF
def tfidf_search(query):
    corpus = df["Title"] + ". " + df["Ingredients"] + ". " + df["Steps"]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, tfidf_matrix).flatten()
    idx = np.argmax(sim)
    if sim[idx] > 0.1:
        return format_recipe(df.iloc[idx])
    return "❌ Tidak ditemukan resep yang relevan."

# Load Gemini (jika tersedia)
@st.cache_resource
def load_gemini(api_key):
    memory = ConversationBufferMemory()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    return ConversationChain(llm=llm, memory=memory)

# UI Streamlit
st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

use_gemini = False
if USE_GEMINI:
    key = st.text_input("🔑 Masukkan API Key Gemini (opsional):", type="password")
    if key:
        try:
            gemini = load_gemini(key)
            use_gemini = True
            st.success("✅ Gemini aktif!")
        except:
            st.warning("❌ API Key salah atau limit habis.")

if "chat" not in st.session_state:
    st.session_state.chat = [{"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"}]

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan resep, bahan, atau masakan..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if use_gemini:
                response = gemini.run(prompt)
            else:
                response = tfidf_search(prompt)
        except Exception as e:
            response = f"❌ Error: {str(e)}"

        st.markdown(response)
        st.session_state.chat.append({"role": "assistant", "content": response})
