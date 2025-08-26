# Movie Search Assignment

This repository contains my solution for the semantic search on movie plots assignment.
This repository demonstrates a semantic search system for movies using sentence embeddings. It allows users to search for movies by plot description and returns the most relevant results.

## Files

- `movie_search.py`: Main Python module containing the movie search logic using Sentence Transformers and pandas.
- `movies.csv`: Sample dataset of movies with titles and plot descriptions.
- `requirements.txt`: List of required Python packages.
- `tests/test_movie_search.py`: Unit tests for the search functionality.
- `solution.ipynb`: Jupyter notebook for interactive exploration and demonstration.

## Setup

1. **Install dependencies**  
   Run the following command in your environment:

   ```
   pip install -r requirements.txt
   ```

2. **Run the notebook**  
   Open `solution.ipynb` in Jupyter and follow the instructions.

3. **Run tests**  
   Execute the tests to verify functionality:
   ```
   python -m unittest discover -s tests -p "test_*.py" -v
   ```

## Usage

- Use the `search_movies(query, top_n)` function in `movie_search.py` to find the top N movies most relevant to your query.
- The function returns a pandas DataFrame with the movie title, plot, and similarity score.

## Example

```python
from movie_search import search_movies
results = search_movies("spy thriller in Paris", top_n=3)
print(results)
```

## Requirements

- pandas
- sentence-transformers
- scikit-learn
- torch

## License

For educational use only.
