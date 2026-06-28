from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# In-memory document store
documents = [
    "Python is a high-level programming language.",
    "LLMs are trained on vast amounts of text data.",
    "RAG combines retrieval and generation to answer questions.",
    "Embeddings map text to dense vectors for semantic search.",
    "Groq provides extremely fast LLM inference.",
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all documents
doc_embeddings = model.encode(documents)

def retrieve(query: str) -> str:
    """Find the most similar document to the query."""
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    best_index = scores.argmax()
    return documents[best_index]

def ask(query: str) -> str:
    """Retrieve context and generate a grounded answer."""
    context = retrieve(query)
    print(f"\nRetrieved context: {context}")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": (
                "Answer the user's question using ONLY the following context. "
                f"If the answer is not in the context, say 'I don't know.'\n\nContext: {context}"
            )},
            {"role": "user", "content": query}
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# Test it
query = input("Query: ")
answer = ask(query)
print(f"\nAnswer: {answer}")