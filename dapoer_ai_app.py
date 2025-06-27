pip install scikit-learn
pip install streamlit pandas scikit-learn langchain langchain-google-genai
# ===== AUTO-INSTALL JIKA BELUM ADA =====
import os
import subprocess
import importlib.util

def install(package):
    subprocess.call(["pip", "install", package])

# Paket wajib
for pkg in ["streamlit", "pandas", "scikit-learn"]:
    if importlib.util.find_spec(pkg) is None:
        install(pkg)

# Paket opsional (Gemini)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    USE_GEMINI = True
except ImportError:
    USE_GEMINI = False
    try:
        install("langchain-google-genai")
        install("langchain")
        USE_GEMINI = True
    except:
        pass

# ====== APP LOGIC ======
import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional Gemini
if USE_GEMINI:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/audreeynr/dapoer-ai/main/data/Indonesian_Food_Recipes.csv"
    df = pd.read_csv(url).dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()
    df["Title_Norm"] = df["Title"].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    df["Ingredients_Norm"] = df["Ingredients"].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    df["Steps_Norm"] = df["Steps"].str.lower().str.replace(r'[^a-z0-9\s]', '', regex=True)
    return df

df = load_data()

def format_recipe(row):
    bahan_list = [f"- {b.strip().capitalize()}" for b in re.split(r'\n|,|--', row['Ingredients']) if b.strip()]
    return f"""🍽 **{row['Title']}**

**Bahan-bahan:**  
{chr(10).join(bahan_list)}

**Langkah-langkah:**  
{row['Steps'].strip()}"""

def search_by_title(query):
    q = re.sub(r'[^a-z0-9\s]', '', query.lower())
    match = df[df["Title_Norm"].str.contains(q)]
    return format_recipe(match.iloc[0]) if not match.empty else "❌ Resep tidak ditemukan."

def search_by_ingredient(query):
    q = re.sub(r'[^a-z0-9\s]', '', query.lower())
    keywords = [w for w in q.split() if len(w) > 2]
    result = df[df["Ingredients_Norm"].apply(lambda x: all(k in x for k in keywords))]
    if not result.empty:
        return "🍳 Masakan dengan bahan tersebut:\n- " + "\n- ".join(result["Title"].head(5))
    return "❌ Tidak ditemukan masakan dengan bahan itu."

def search_by_method(query):
    q = query.lower()
    for method in ["goreng", "rebus", "panggang", "kukus"]:
        if method in q:
            result = df[df["Steps_Norm"].str.contains(method)]
            if not result.empty:
                return f"🔥 Masakan dengan cara {method}:\n- " + "\n- ".join(result["Title"].head(5))
    return "❌ Tidak ditemukan metode memasak."

def recommend_easy():
    short_steps = df[df["Steps"].str.len() < 300]
    return "✨ Rekomendasi masakan mudah:\n- " + "\n- ".join(short_steps["Title"].head(5))

def tfidf_search(query):
    corpus = df["Title"] + ". " + df["Ingredients"] + ". " + df["Steps"]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    q_vec = TfidfVectorizer().fit(corpus).transform([query])
    sims = cosine_similarity(q_vec, vectorizer).flatten()
    idx = np.argmax(sims)
    if sims[idx] > 0.1:
        return format_recipe(df.iloc[idx])
    return "❌ Tidak ditemukan hasil yang mirip."

@st.cache_resource
def load_gemini(api_key):
    memory = ConversationBufferMemory()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    return ConversationChain(llm=llm, memory=memory)

# ==== STREAMLIT UI ====
st.set_page_config(page_title="Dapoer-AI Hybrid", page_icon="🍲")
st.title("🍛 Dapoer-AI (Offline + Opsional Gemini)")

use_gemini = False
if USE_GEMINI:
    api_key = st.text_input("🔑 Masukkan Gemini API Key (opsional):", type="password")
    if api_key:
        try:
            gemini = load_gemini(api_key)
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
        response = ""
        try:
            if "judul" in prompt.lower():
                response = search_by_title(prompt)
            elif "bahan" in prompt.lower():
                response = search_by_ingredient(prompt)
            elif any(m in prompt.lower() for m in ["goreng", "panggang", "kukus", "rebus"]):
                response = search_by_method(prompt)
            elif "mudah" in prompt.lower() or "pemula" in prompt.lower():
                response = recommend_easy()
            elif use_gemini:
                response = gemini.run(prompt)
            else:
                response = tfidf_search(prompt)
        except Exception as e:
            response = f"❌ Error: {e}"

        st.markdown(response)
        st.session_state.chat.append({"role": "assistant", "content": response})
