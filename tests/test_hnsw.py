import numpy as np
from rust_annie import PyHnswIndex

def test_hnsw_basic():
    dim = 64
    index = PyHnswIndex(dims=dim)
    
    # Generate sample data
    data = np.random.rand(1000, dim).astype(np.float32)
    ids = np.arange(1000, dtype=np.int64)
    
    # Add to index
    index.add(data, ids)
    
    # Query
    query = np.random.rand(dim).astype(np.float32)
    ids, dists = index.search(query, k=10)
    
    assert len(ids) == 10
    assert len(dists) == 10
    assert all(isinstance(i, (int, np.integer)) for i in ids)
    assert all(isinstance(d, (float, np.floating)) for d in dists)