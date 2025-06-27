import streamlit as st
import pandas as pd
import re
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

# Load data
CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/refs/heads/main/data/Indonesian_Food_Recipes.csv'
st.write("🔍 Load CSV dari:", CSV_FILE_PATH)
df = pd.read_csv(CSV_FILE_PATH)
df = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

# Normalisasi teks
def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

df['Title_Normalized'] = df['Title'].apply(normalize_text)
df['Ingredients_Normalized'] = df['Ingredients'].apply(normalize_text)
df['Steps_Normalized'] = df['Steps'].apply(normalize_text)

# Format resep
def format_recipe(row):
    bahan_list = [b.strip().capitalize() for b in re.split(r'\n|--|,', row['Ingredients']) if b.strip()]
    bahan_md = "\n".join(f"- {b}" for b in bahan_list)
    langkah_md = row['Steps'].strip()
    return f"""🍽 {row['Title']}\n\n**Bahan-bahan:**\n{bahan_md}\n\n**Langkah Memasak:**\n{langkah_md}"""

# Tools
def search_by_title(query):
    query = normalize_text(query)
    match = df[df['Title_Normalized'].str.contains(query)]
    return format_recipe(match.iloc[0]) if not match.empty else "❌ Resep tidak ditemukan berdasarkan judul."

def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    keywords = [w for w in normalize_text(query).split() if w not in stopwords and len(w) > 2]
    if keywords:
        mask = df['Ingredients_Normalized'].apply(lambda x: all(k in x for k in keywords))
        match = df[mask]
        return "🍽 Masakan dengan bahan tersebut:\n- " + "\n- ".join(match.head(5)['Title']) if not match.empty else "❌ Tidak ditemukan masakan dengan bahan itu."
    return "⚠️ Masukkan bahan yang lebih spesifik."

def search_by_method(query):
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in normalize_text(query):
            match = df[df['Steps_Normalized'].str.contains(metode)]
            return f"🔥 Masakan dimasak dengan cara {metode}:\n- " + "\n- ".join(match.head(5)['Title']) if not match.empty else f"❌ Tidak ditemukan masakan dengan metode {metode}."
    return "⚠️ Tidak ada metode masak yang cocok ditemukan."

def recommend_easy_recipes(query):
    if any(k in normalize_text(query) for k in ['mudah', 'pemula', 'cepat']):
        match = df[df['Steps'].str.len() < 300]
        return "👍 Rekomendasi masakan mudah:\n- " + "\n- ".join(match.head(5)['Title'])
    return "❌ Tidak ditemukan masakan mudah yang relevan."

def create_agent(api_key):
    # Coba 3x kalau error 429
    for _ in range(3):
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7
            )
            tools = [
                Tool(name="SearchByTitle", func=search_by_title, description="Cari resep berdasarkan judul masakan."),
                Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari masakan berdasarkan bahan."),
                Tool(name="SearchByMethod", func=search_by_method, description="Cari masakan berdasarkan metode memasak."),
                Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi masakan yang mudah dibuat.")
            ]
            memory = ConversationBufferMemory(memory_key="chat_history")
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent="zero-shot-react-description",
                memory=memory,
                verbose=False
            )
            return agent
        except Exception as e:
            if "429" in str(e):
                time.sleep(20)  # tunggu 20 detik terus coba lagi
            else:
                raise e
    raise Exception("Gagal membuat agent setelah 3 kali percobaan karena quota limit.")

# ================== STREAMLIT ==================

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

api_key = st.text_input("🔑 Masukkan API Key Gemini kamu:", type="password")
if not api_key:
    st.warning("Masukkan API key dulu biar bisa jalanin aplikasi.")
    st.stop()

@st.cache_resource
def get_agent(api_key): return create_agent(api_key)
agent = get_agent(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan resep, bahan, atau metode memasak..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"⚠️ Terjadi error: {str(e)}"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
