# dapoer_module.py
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

CSV_FILE_PATH = "https://raw.githubusercontent.com/audreeynr/dapoer-ai/main/data/Indonesian_Food_Recipes.csv"

df = pd.read_csv(CSV_FILE_PATH).dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

df['Title_Norm'] = df['Title'].apply(normalize_text)
df['Ingredients_Norm'] = df['Ingredients'].apply(normalize_text)
df['Steps_Norm'] = df['Steps'].apply(normalize_text)

def format_recipe(row):
    bahan = "\n".join(f"- {x.strip().capitalize()}" for x in re.split(r'\n|,|--', row['Ingredients']) if x.strip())
    langkah = row['Steps'].strip()
    return f"🍽 {row['Title']}\n\n**Bahan:**\n{bahan}\n\n**Langkah:**\n{langkah}"

def search_by_title(query):
    q = normalize_text(query)
    match = df[df['Title_Norm'].str.contains(q)]
    return format_recipe(match.iloc[0]) if not match.empty else "❌ Tidak ditemukan resep dengan judul itu."

def search_by_ingredients(query):
    keywords = [w for w in normalize_text(query).split() if len(w) > 2]
    match = df[df['Ingredients_Norm'].apply(lambda x: all(k in x for k in keywords))]
    if not match.empty:
        return "📌 Masakan:\n" + "\n".join("- " + t for t in match['Title'].head(5))
    return "❌ Tidak ada masakan cocok."

def search_by_method(query):
    for metode in ['goreng', 'rebus', 'kukus', 'panggang']:
        if metode in normalize_text(query):
            match = df[df['Steps_Norm'].str.contains(metode)]
            return f"🔥 Metode {metode.title()}:\n" + "\n".join("- " + t for t in match['Title'].head(5))
    return "❌ Metode masak tidak ditemukan."

def recommend_easy_recipes(query):
    if "mudah" in normalize_text(query):
        match = df[df['Steps'].str.len() < 300]
        return "👍 Rekomendasi masakan mudah:\n" + "\n".join("- " + t for t in match['Title'].head(5))
    return "❌ Tidak ada rekomendasi yang cocok."

def build_vectorstore(api_key):
    docs = [Document(page_content=f"{r['Title']}\n{r['Ingredients']}\n{r['Steps']}") for _, r in df.iterrows()]
    texts = CharacterTextSplitter(chunk_size=300, chunk_overlap=20).split_documents(docs)
    return FAISS.from_documents(texts, GoogleGenerativeAIEmbeddings(google_api_key=api_key))

def rag_search(api_key, query):
    retriever = build_vectorstore(api_key).as_retriever()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "❌ Tidak ada info relevan."
    return "\n\n".join(d.page_content for d in docs[:3])

def create_agent(api_key):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)

    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep dari judul masakan."),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari berdasarkan bahan."),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari berdasarkan metode."),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi untuk pemula."),
        Tool(name="RAGSearch", func=lambda q: rag_search(api_key, q), description="Cari resep pakai vectorstore RAG.")
    ]

    return initialize_agent(tools, llm, agent="zero-shot-react-description", memory=ConversationBufferMemory())
    memory = ConversationBufferMemory(memory_key="chat_history")
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", memory=memory, verbose=False)
    return agent
