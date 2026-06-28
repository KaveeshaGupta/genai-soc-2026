from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get input from user
sentence_a = input("Sentence A: ")
sentence_b = input("Sentence B: ")

# Generate embeddings
embedding_a = model.encode([sentence_a])
embedding_b = model.encode([sentence_b])

# Compute cosine similarity
score = cosine_similarity(embedding_a, embedding_b)[0][0]

print(f"\nCosine Similarity: {score:.2f}")