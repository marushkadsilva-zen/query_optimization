# ==============================
# IMPORTS
# ==============================
import os
from dotenv import load_dotenv
from google import genai

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==============================
# LOAD ENV
# ==============================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY missing")

client = genai.Client(api_key=api_key)

# ==============================
# GEMINI FUNCTION
# ==============================
def ask_gemini(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"


# ==============================
# LOAD DATA
# ==============================
loader = TextLoader("data.txt")
documents = loader.load()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(documents)

# ==============================
# VECTOR DB
# ==============================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(docs, embedding)
retriever = db.as_retriever(search_kwargs={"k": 3})


# ==============================
# QUERY OPTIMIZATION
# ==============================

def rewrite_query(query):
    return ask_gemini(f"Rewrite clearly:\n{query}")


def multi_query(query):
    result = ask_gemini(f"""
Generate 3 clean search queries.
Do NOT add numbering or explanation.
Only return plain queries.

Query: {query}
""")

    return [q.strip("- ").strip() for q in result.split("\n") if q.strip()]


def hyde_query(query):
    return ask_gemini(f"Write a detailed answer:\n{query}")

def decompose_query(query):
    result = ask_gemini(f"""
Break into simple sub-questions.
No explanation. No numbering.

Query: {query}
""")

    return [q.strip("- ").strip() for q in result.split("\n") if q.strip()]


# ==============================
# RETRIEVAL
# ==============================
def retrieve(query):
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


# ==============================
# PIPELINE
# ==============================
def run_pipeline(query):
    print("\n🔹 ORIGINAL:", query)

    rewritten = rewrite_query(query)
    print("\n🔹 REWRITTEN:", rewritten)

    multi = multi_query(rewritten)
    print("\n🔹 MULTI:", multi)

    hyde = hyde_query(rewritten)
    print("\n🔹 HYDE:", hyde[:200], "...")

    sub = decompose_query(query)
    print("\n🔹 SUB:", sub)

    all_queries = [rewritten] + multi + sub + [hyde]

    results = []
    for q in all_queries:
        results.extend(retrieve(q))

    results = list(set(results))

    print("\n🔹 RESULTS:")
    for r in results:
        print("-", r)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    while True:
        q = input("\nEnter query (exit to stop): ")
        if q.lower() == "exit":
            break
        run_pipeline(q)