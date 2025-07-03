# AnnIndex - Brute-force Nearest Neighbor Search

The `AnnIndex` class provides efficient brute-force nearest neighbor search with support for multiple distance metrics.

## Constructor

### `AnnIndex(dim: int, metric: Distance)`
Creates a new brute-force index.

- `dim` (int): Vector dimension
- `metric` (Distance): Distance metric (`EUCLIDEAN`, `COSINE`, `MANHATTAN`, `CHEBYSHEV`)

### `new_minkowski(dim: int, p: float)`
Creates a Minkowski distance index.

- `dim` (int): Vector dimension
- `p` (float): Minkowski exponent (p > 0)

## Methods

### `add(data: ndarray, ids: ndarray)`
Add vectors to the index.

- `data`: N×dim array of float32 vectors
- `ids`: N-dimensional array of int64 IDs

### `search(query: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Search for k nearest neighbors.

- `query`: dim-dimensional query vector
- `k`: Number of neighbors to return
- Returns: (neighbor IDs, distances)

### `search_batch(queries: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Batch search for multiple queries.

- `queries`: M×dim array of queries
- `k`: Number of neighbors per query
- Returns: (M×k IDs, M×k distances)

### `search_filter_py(query: ndarray, k: int, filter_fn: Callable[[int], bool]) -> Tuple[ndarray, ndarray]`
Search with ID filtering.

- `query`: dim-dimensional query vector
- `k`: Maximum neighbors to return
- `filter_fn`: Function that returns True for allowed IDs
- Returns: (filtered IDs, filtered distances)

### `save(path: str)`
Save index to disk.

### `static load(path: str) -> AnnIndex`
Load index from disk.

## Example
```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)
```