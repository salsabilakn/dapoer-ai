import streamlit as st
import pandas as pd
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

# --- SETUP FILE CSV ---
CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/refs/heads/main/data/Indonesian_Food_Recipes.csv'

# Load data
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

# Format output resep
def format_recipe(row):
    bahan = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan if b.strip()]
    langkah = row['Steps'].strip()
    return f"""🍽 **{row['Title']}**

**Bahan-bahan:**  
- {"\n- ".join(bahan_list)}

**Langkah Memasak:**  
{langkah}"""

# Tool 1: Berdasarkan judul
def search_by_title(query):
    q = normalize_text(query)
    match = df[df['Title_Normalized'].str.contains(q)]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "❌ Resep tidak ditemukan berdasarkan judul."

# Tool 2: Berdasarkan bahan
def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    tokens = [w for w in normalize_text(query).split() if w not in stopwords and len(w) > 2]
    mask = df['Ingredients_Normalized'].apply(lambda x: all(k in x for k in tokens))
    hasil = df[mask]
    if not hasil.empty:
        return "Masakan dengan bahan tersebut:\n- " + "\n- ".join(hasil.head(5)['Title'].tolist())
    return "❌ Tidak ditemukan masakan dengan bahan tersebut."

# Tool 3: Berdasarkan metode masak
def search_by_method(query):
    q = normalize_text(query)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in q:
            cocok = df[df['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                return f"Masakan dimasak dengan cara {metode}:\n- " + "\n- ".join(cocok.head(5)['Title'].tolist())
    return "❌ Tidak ditemukan metode memasak yang cocok."

# Tool 4: Masakan mudah
def recommend_easy_recipes(query):
    if "mudah" in query.lower() or "pemula" in query.lower():
        hasil = df[df['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    return "❌ Tidak ditemukan masakan mudah yang cocok."

# Agent Langchain
def create_agent(api_key):
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

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=False
    )

# --- STREAMLIT UI ---
st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# API Key input
GOOGLE_API_KEY = st.text_input("Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Silakan masukkan API key untuk mulai.")
    st.stop()

agent = create_agent(GOOGLE_API_KEY)

# Inisialisasi chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"})

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat
if prompt := st.chat_input("Tanyakan resep, bahan, atau metode memasak..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = agent.run(prompt)
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                response = "⚠️ Kuota API kamu udah habis. Coba lagi besok atau pakai API Key lain."
            else:
                response = f"❌ Error: {e}"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
