import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("vector_store/index.faiss")

# Store document chunks and their sources
docs = []
sources = []

# Chunk settings
chunk_size = 500
overlap = 100

# Load documents and create chunks
for file in os.listdir("docs"):

    path = os.path.join("docs", file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    for i in range(0, len(text), chunk_size - overlap):

        chunk = text[i:i + chunk_size]

        docs.append(chunk)
        sources.append(file)


def search(question, top_k=3):

    # Convert question to embedding
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding)

    # Search FAISS index
    distances, indices = index.search(query_embedding, top_k)

    results = []
    threshold = 1.5   # relevance threshold

    for i in range(top_k):

        chunk_id = indices[0][i]
        score = distances[0][i]

        # Ignore invalid index
        if chunk_id < 0 or chunk_id >= len(docs):
            continue

        # Ignore irrelevant results
        if score > threshold:
            continue

        results.append({
            "document": sources[chunk_id],
            "snippet": docs[chunk_id],
            "score": float(score)
        })

    return results


# CLI testing
if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        results = search(question)

        if len(results) == 0:
            print("\nNo relevant information found.")

        else:

            print("\nTop Results:\n")

            for r in results:

                print("Document:", r["document"])
                print("Snippet:", r["snippet"])
                print("Score:", r["score"])
                print("-----")