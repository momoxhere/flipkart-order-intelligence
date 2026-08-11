# part3/build_index.py
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_vector_index():
    print("Loading policies...")
    with open("part3/knowledge_base/policies.json", "r") as f:
        policies = json.load(f)

    chunks = []
    # Sentence-level chunking
    for doc in policies:
        sentences = [s.strip() + "." for s in doc["text"].split(".") if len(s.strip()) > 5]
        for i, sentence in enumerate(sentences):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_C{i+1:02d}",
                "document_id": doc["doc_id"],
                "text": sentence
            })

    print(f"Created {len(chunks)} sentence-wise chunks. Loading embedding model...")
    # Using free local sentence-transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embedding_dim = embeddings.shape[1]

    print("Building FAISS index (Cosine Similarity)...")
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.array(embeddings).astype('float32'))

    # Save index and metadata
    faiss.write_index(index, "vector_index/faiss.index")
    with open("vector_index/chunk_metadata.json", "w") as f:
        json.dump(chunks, f, indent=2)
        
    print("Vector index built and saved successfully to vector_index/")

if __name__ == "__main__":
    build_vector_index()