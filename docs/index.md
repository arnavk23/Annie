# Annie Documentation

Blazingly fast Approximate Nearest Neighbors in Rust

## Installation
```bash
pip install rust_annie
```

## Basic Usage
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

## Key Features

- Multiple distance metrics
- CPU/GPU acceleration
- Thread-safe indexes
- Filtered search
- HNSW support