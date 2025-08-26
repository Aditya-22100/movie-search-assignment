import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# -------------------------------
# Load Data
# -------------------------------
try:
    # Build path to movies.csv (must be in same directory as this script)
    csv_path = os.path.join(os.path.dirname(__file__), "movies.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    
    # Load CSV into DataFrame
    df = pd.read_csv(csv_path)
    
    if "plot" not in df.columns:
        raise ValueError("CSV must contain a 'plot' column for embeddings.")

except Exception as e:
    raise SystemExit(f"Error loading CSV: {e}")


# -------------------------------
# Load Sentence Transformer Model
# -------------------------------
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    raise SystemExit(f"Error loading model: {e}")


# -------------------------------
# Precompute Embeddings
# -------------------------------
try:
    embeddings = model.encode(
        df["plot"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True
    )
except Exception as e:
    raise SystemExit(f"Error generating embeddings: {e}")


# -------------------------------
# Search Function
# -------------------------------
def search_movies(query: str, top_n: int = 5):
    """
    Search for movies based on semantic similarity of plots.

    Args:
        query (str): User query describing movie plot/content.
        top_n (int): Number of top results to return.

    Returns:
        pd.DataFrame: Matching movies with similarity scores.
    """
    # Validate inputs
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    
    try:
        # Encode query
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Compute cosine similarities
        sims = util.cos_sim(q_emb, embeddings).cpu().numpy().flatten()
        
        # Get top results
        idx = sims.argsort()[::-1][:top_n]
        
        # Return selected rows with similarity scores
        return df.iloc[idx].assign(similarity=sims[idx])
    
    except Exception as e:
        raise RuntimeError(f"Error during search: {e}")
