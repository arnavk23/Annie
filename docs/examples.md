# Annie Examples

## Basic Usage
```python
import numpy as np
from rust_annie import AnnIndex, Distance

# Create index
index = AnnIndex(128, Distance.EUCLIDEAN)

# Generate and add data
data = np.random.rand(1000, 128).astype(np.float32)
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Single query
query = np.random.rand(128).astype(np.float32)
neighbor_ids, distances = index.search(query, k=5)

# Batch queries
queries = np.random.rand(10, 128).astype(np.float32)
batch_ids, batch_dists = index.search_batch(queries, k=3)
```

## Filtered Search
```python
# Create index with sample data
index = AnnIndex(3, Distance.EUCLIDEAN)
data = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
], dtype=np.float32)
ids = np.array([10, 20, 30], dtype=np.int64)
index.add(data, ids)

# Define filter function
def even_ids(id: int) -> bool:
    return id % 2 == 0

# Filtered search
query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
filtered_ids, filtered_dists = index.search_filter_py(query, k=3, filter_fn=even_ids)
# Only IDs 10 and 30 will be returned (20 is odd)
```

## HNSW Index
```python
from rust_annie import PyHnswIndex

# Create HNSW index
index = PyHnswIndex(dims=128)

# Add large dataset
data = np.random.rand(100000, 128).astype(np.float32)
ids = np.arange(100000, dtype=np.int64)
index.add(data, ids)

# Fast approximate search
query = np.random.rand(128).astype(np.float32)
neighbor_ids, _ = index.search(query, k=10)
```

## Saving and Loading
```python
# Create and save index
index = AnnIndex(64, Distance.COSINE)
data = np.random.rand(500, 64).astype(np.float32)
ids = np.arange(500, dtype=np.int64)
index.add(data, ids)
index.save("my_index")

# Load index
loaded_index = AnnIndex.load("my_index")
```

## Thread-safe Operations
```python
from rust_annie import ThreadSafeAnnIndex, Distance
from concurrent.futures import ThreadPoolExecutor

index = ThreadSafeAnnIndex(256, Distance.MANHATTAN)

# Concurrent writes
with ThreadPoolExecutor() as executor:
    for i in range(10):
        data = np.random.rand(100, 256).astype(np.float32)
        ids = np.arange(i*100, (i+1)*100, dtype=np.int64)
        executor.submit(index.add, data, ids)

# Concurrent reads
with ThreadPoolExecutor() as executor:
    futures = []
    for _ in range(100):
        query = np.random.rand(256).astype(np.float32)
        futures.append(executor.submit(index.search, query, k=3))
    
    results = [f.result() for f in futures]
```

## Minkowski Distance
```python
# Create index with custom distance
index = AnnIndex.new_minkowski(dim=64, p=2.5)
data = np.random.rand(200, 64).astype(np.float32)
ids = np.arange(200, dtype=np.int64)
index.add(data, ids)

# Search with Minkowski distance
query = np.random.rand(64).astype(np.float32)
ids, dists = index.search(query, k=5)
```