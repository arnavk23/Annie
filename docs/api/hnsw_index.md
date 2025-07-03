# PyHnswIndex - Approximate Nearest Neighbors with HNSW

The `PyHnswIndex` class provides approximate nearest neighbor search using Hierarchical Navigable Small World (HNSW) graphs.

## Constructor

### `PyHnswIndex(dims: int)`
Creates a new HNSW index.

- `dims` (int): Vector dimension

## Methods

### `add(data: ndarray, ids: ndarray)`
Add vectors to the index.

- `data`: N×dims array of float32 vectors
- `ids`: N-dimensional array of int64 IDs

### `search(vector: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Search for k approximate nearest neighbors.

- `vector`: dims-dimensional query vector
- `k`: Number of neighbors to return
- Returns: (neighbor IDs, distances)

### `save(path: str)`
Save index to disk.

### `static load(path: str) -> PyHnswIndex`
Load index from disk (currently not implemented)

## Example
```python
import numpy as np
from rust_annie import PyHnswIndex

# Create index
index = PyHnswIndex(dims=128)

# Add data
data = np.random.rand(10000, 128).astype(np.float32)
ids = np.arange(10000, dtype=np.int64)
index.add(data, ids)

# Search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, _ = index.search(query, k=10)
```