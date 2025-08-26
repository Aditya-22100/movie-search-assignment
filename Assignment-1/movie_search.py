
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load movies.csv from the same directory as this file
csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")
df = pd.read_csv(csv_path)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(
    df["plot"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True
)


def search_movies(query, top_n=5):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = util.cos_sim(q_emb, embeddings).cpu().numpy().flatten()
    idx = sims.argsort()[::-1][:top_n]
    return df.iloc[idx].assign(similarity=sims[idx])
