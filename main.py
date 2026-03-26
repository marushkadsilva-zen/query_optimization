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
# def ask_gemini(prompt):
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=prompt
#         )
#         return response.text.strip()
#     except Exception as e:
#         return f"Error: {e}"

def ask_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            # ✅ Check if response is valid
            if hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                return ""

        except Exception as e:
            print(f"⚠️ Gemini Error (attempt {attempt+1}): {e}")

    return None  # ✅ return None instead of error string


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

# 🔹 Rewrite Query
def rewrite_query(query):
    return ask_gemini(f"""
Rewrite into ONE short search query (max 10 words).
Keep meaning SAME.

Query: {query}
""")

# 🔹 Multi Query (FIXED)
def multi_query(query):
    result = ask_gemini(f"""
You are an expert in search query generation.

Generate EXACTLY 3 alternative search queries that are:
- Closely related to the original query
- Same topic ONLY
- Different wording, same meaning

STRICT RULES:
- DO NOT change topic
- DO NOT introduce unrelated concepts
- No numbering, no explanation
- One query per line

Original Query: {query}
""")

    print("\n🔍 DEBUG MULTI RAW:\n", result)

    return [q.strip("- ").strip() for q in result.split("\n") if q.strip()]

# 🔹 HyDE
def hyde_query(query):
    return ask_gemini(f"""
Write a detailed answer for:
{query}
""")

# 🔹 Decomposition
def decompose_query(query):
    result = ask_gemini(f"""
Break into simple sub-questions.
No explanation. No numbering.

Query: {query}
""")

    return [q.strip("- ").strip() for q in result.split("\n") if q.strip()]

# 🔹 Step-Back Prompting
def step_back_query(query):
    return ask_gemini(f"""
Generate a broader question for better context retrieval.

Rules:
- Keep it simple
- Same topic
- One question only

Query: {query}
""").strip()

# ==============================
# FILTERING (IMPORTANT)
# ==============================
def filter_queries(original, queries):
    keywords = set(original.lower().split())

    filtered = []
    for q in queries:
        if any(word in q.lower() for word in keywords):
            filtered.append(q)

    return filtered

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
    print("\n==============================")
    print("🔹 ORIGINAL:", query)

    # Rewrite
    rewritten = rewrite_query(query)
    print("\n🔹 REWRITTEN:", rewritten)

    # Step-back
    step_back = step_back_query(query)
    print("\n🔹 STEP-BACK:", step_back)

    # Multi-query (FIXED)
    multi = multi_query(query)   # ✅ USE ORIGINAL
    multi = filter_queries(query, multi)
    print("\n🔹 MULTI (FILTERED):", multi)

    # HyDE
    hyde = hyde_query(query)
    print("\n🔹 HYDE:", hyde[:200], "...")

    # Decomposition
    sub = decompose_query(query)
    print("\n🔹 SUB:", sub)

    # ==============================
    # COMBINE QUERIES
    # ==============================
    all_queries = (
        [rewritten] +
        [step_back] +
        multi +
        sub[:2] +
        [hyde]
    )

    # ==============================
    # RETRIEVE
    # ==============================
    results = []
    for q in all_queries:
        results.extend(retrieve(q))

    # Remove duplicates
    results = list(set(results))

    print("\n🔹 RESULTS:")
    for r in results:
        print("-", r)

    print("==============================\n")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    while True:
        q = input("\nEnter query (exit to stop): ")
        if q.lower() == "exit":
            break
        run_pipeline(q)