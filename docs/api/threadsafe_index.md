# ThreadSafeAnnIndex - Thread-safe Nearest Neighbor Index

The `ThreadSafeAnnIndex` class provides a thread-safe wrapper around `AnnIndex` for concurrent access.

## Constructor

### `ThreadSafeAnnIndex(dim: int, metric: Distance)`
Creates a new thread-safe index.

- `dim` (int): Vector dimension
- `metric` (Distance): Distance metric

## Methods

### `add(data: ndarray, ids: ndarray)`
Thread-safe vector addition.

### `remove(ids: List[int])`
Thread-safe removal by IDs.

### `search(query: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Thread-safe single query search.

### `search_batch(queries: ndarray, k: int) -> Tuple[ndarray, ndarray]`
Thread-safe batch search.

### `save(path: str)`
Thread-safe save.

### `static load(path: str) -> ThreadSafeAnnIndex`
Thread-safe load.

## Example
```python
import numpy as np
from rust_annie import ThreadSafeAnnIndex, Distance
from concurrent.futures import ThreadPoolExecutor

# Create index
index = ThreadSafeAnnIndex(128, Distance.COSINE)

# Add data from multiple threads
with ThreadPoolExecutor() as executor:
    for i in range(4):
        data = np.random.rand(250, 128).astype(np.float32)
        ids = np.arange(i*250, (i+1)*250, dtype=np.int64)
        executor.submit(index.add, data, ids)

# Concurrent searches
with ThreadPoolExecutor() as executor:
    futures = []
    for _ in range(10):
        query = np.random.rand(128).astype(np.float32)
        futures.append(executor.submit(index.search, query, k=5))
    
    for future in futures:
        ids, dists = future.result()
```