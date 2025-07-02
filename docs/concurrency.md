# Using `ThreadSafeAnnIndex` and `PyHnswIndex` for Concurrent Access

Annie exposes a thread-safe version of its ANN index (`AnnIndex`) for use in Python. This is useful when you want to perform parallel search or update operations from Python threads. Additionally, the `PyHnswIndex` class provides a Python interface to the HNSW index, which now includes enhanced data handling capabilities.

## Key Features

- Safe concurrent read access (`search`, `search_batch`)
- Exclusive write access (`add`, `remove`)
- Backed by Rust `RwLock` and exposed via PyO3
- `PyHnswIndex` supports mapping internal IDs to user IDs and handling vector data efficiently

## Example

```python
from annie import ThreadSafeAnnIndex, Distance
import numpy as np
import threading

# Create index
index = ThreadSafeAnnIndex(128, Distance.Cosine)

# Add vectors
data = np.random.rand(1000, 128).astype('float32')
ids = np.arange(1000, dtype=np.int64)
index.add(data, ids)

# Run concurrent searches
def run_search():
    query = np.random.rand(128).astype('float32')
    ids, distances = index.search(query, 10)
    print(ids)

threads = [threading.Thread(target=run_search) for _ in range(4)]
[t.start() for t in threads]
[t.join() for t in threads]

# Using PyHnswIndex
from rust_annie import PyHnswIndex

# Create HNSW index
hnsw_index = PyHnswIndex(dims=128)

# Add vectors to HNSW index
hnsw_index.add(data, ids)

# Search in HNSW index
query = np.random.rand(128).astype('float32')
user_ids, distances = hnsw_index.search(query, 10)
print(user_ids)
```

# CI/CD Pipeline for PyPI Publishing

The CI/CD pipeline for PyPI publishing has been updated to include parallel jobs for building wheels and source distributions across multiple operating systems and Python versions. This involves concurrency considerations that should be documented for users who are integrating or maintaining the pipeline.

## Pipeline Overview

The pipeline is triggered on pushes and pull requests to the `main` branch, as well as manually via `workflow_dispatch`. It includes the following jobs:

- **Test**: Runs on `ubuntu-latest` and includes steps for checking out the code, setting up Rust, caching dependencies, running tests, and checking code formatting.
- **Build Wheels**: Runs in parallel across `ubuntu-latest`, `windows-latest`, and `macos-latest` for Python versions 3.8, 3.9, 3.10, and 3.11. This job builds the wheels using `maturin` and uploads them as artifacts.
- **Build Source Distribution**: Runs on `ubuntu-latest` and builds the source distribution using `maturin`, uploading it as an artifact.
- **Publish to TestPyPI**: Publishes the built artifacts to TestPyPI if triggered via `workflow_dispatch` with the appropriate input.
- **Publish to PyPI**: Publishes the built artifacts to PyPI if triggered via `workflow_dispatch` with the appropriate input.

## Concurrency Considerations

- **Parallel Builds**: The `build-wheels` job utilizes a matrix strategy to run builds concurrently across different operating systems and Python versions. This reduces the overall build time but requires careful management of dependencies and environment setup to ensure consistency across platforms.
- **Artifact Management**: Artifacts from parallel jobs are downloaded and flattened before publishing to ensure all necessary files are available in a single directory structure for the publish steps.
- **Conditional Publishing**: Publishing steps are conditionally executed based on manual triggers and input parameters, allowing for flexible deployment strategies.

By understanding these concurrency considerations, users can effectively manage and extend the CI/CD pipeline to suit their specific needs.