import numpy as np
from rust_annie import AnnIndex, Distance

index = AnnIndex.new(3, Distance.Euclidean)
data = np.array([
    [0.1, 0.2, 0.3],
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0],
], dtype=np.float32)
ids = np.array([1, 2, 3, 4], dtype=np.int64)

index.add(data, ids)

def even_id_filter(i):
    return i % 2 == 0

query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ids, dists = index.search_filter(query, even_id_filter, k=2)
print("Filtered IDs:", ids)
print("Distances:", dists)
