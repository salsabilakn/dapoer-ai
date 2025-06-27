import pandas as pd
import re
import langchain.vectorstores.faiss  # ⬅️ paksa import modul FAISS
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/main/data/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH).dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()
    return text

df['Title_Normalized'] = df['Title'].apply(normalize_text)
df['Ingredients_Normalized'] = df['Ingredients'].apply(normalize_text)
df['Steps_Normalized'] = df['Steps'].apply(normalize_text)

def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    return f"""🍽 {row['Title']}\n\nBahan-bahan:\n{bahan_md}\n\nLangkah:\n{row['Steps'].strip()}"""

def search_by_title(query):
    q = normalize_text(query)
    hasil = df[df['Title_Normalized'].str.contains(q)]
    return format_recipe(hasil.iloc[0]) if not hasil.empty else "❌ Tidak ditemukan resep dengan judul itu."

def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    q = normalize_text(query)
    keywords = [w for w in q.split() if w not in stopwords and len(w) > 2]
    if keywords:
        hasil = df[df['Ingredients_Normalized'].apply(lambda x: all(k in x for k in keywords))]
        if not hasil.empty:
            return "📋 Masakan yang cocok:\n- " + "\n- ".join(hasil['Title'].head(5).tolist())
    return "❌ Tidak ditemukan resep dengan bahan tersebut."

def search_by_method(query):
    q = normalize_text(query)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in q:
            hasil = df[df['Steps_Normalized'].str.contains(metode)]
            if not hasil.empty:
                return f"🔥 Resep dengan metode {metode}:\n- " + "\n- ".join(hasil['Title'].head(5).tolist())
    return "❌ Tidak ditemukan metode memasak itu."

def recommend_easy_recipes(query):
    if any(k in normalize_text(query) for k in ['mudah', 'pemula']):
        hasil = df[df['Steps'].str.len() < 300]
        return "🧑‍🍳 Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil['Title'].head(5).tolist())
    return "❌ Tidak ada masakan mudah relevan."

def rag_search(api_key, query):
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['Title']}\nIngredients: {row['Ingredients']}\nSteps: {row['Steps']}"
        docs.append(Document(page_content=content))

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key)
    vs = FAISS.from_documents(texts, embeddings)
    retriever = vs.as_retriever()
    found = retriever.get_relevant_documents(query)

    if not found:
        fallback = df.sample(3)
        return "🔁 Tidak ada hasil RAG. Coba ini:\n\n" + "\n\n".join([format_recipe(row) for _, row in fallback.iterrows()])
    return "\n\n".join([doc.page_content for doc in found[:3]])

def create_agent(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep dari nama masakan"),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari masakan dari bahan"),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari dari metode memasak"),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi resep mudah"),
        Tool(name="RAGSearch", func=lambda q: rag_search(api_key, q), description="RAG dari dokumen resep")
    ]
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=False)
    return agent
