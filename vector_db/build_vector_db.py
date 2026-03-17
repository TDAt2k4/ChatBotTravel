import pandas as pd
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("../dataset/vietnam_travel_final.csv")

texts = df["content"].fillna("").tolist()

# =========================
# 2. LOAD MODEL
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 3. CREATE EMBEDDING
# =========================
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)

# normalize (QUAN TRỌNG)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

embeddings = embeddings.astype("float32")

# =========================
# 4. FAISS INDEX
# =========================
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # cosine similarity

index.add(embeddings)

# =========================
# 5. SAVE
# =========================
faiss.write_index(index, "travel_index.faiss")

with open("travel_metadata.pkl", "wb") as f:
    pickle.dump(df.to_dict("records"), f)

# =========================
# 6. DONE
# =========================
print("Vector database created!")
print("Total vectors:", index.ntotal)