import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

docs = []
sources = []

# Read documents
for file in os.listdir("docs"):

    path = os.path.join("docs", file)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    for chunk in chunks:
        docs.append(chunk)
        sources.append(file)

print("Total chunks:", len(docs))

# Create embeddings
embeddings = model.encode(docs)

# Convert to numpy array
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to FAISS
index.add(embeddings)

# Save index
faiss.write_index(index, "vector_store/index.faiss")

print("Vector store created successfully")