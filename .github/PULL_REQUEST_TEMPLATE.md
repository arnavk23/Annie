# 🚀 Pull Request Template

**What does this PR do?**
- [ ] Fixes a bug
- [X] Adds a new feature
- [ ] Improves performance
- [ ] Adds tests
- [X] Updates documentation

**Summary of the changes:**


**Objective:** Create a Jupyter notebook showcasing:
1. **Computational speedup** comparing pure Python vs C++ bindings (using PyBind11)
2. **HNSW integration** for approximate nearest neighbor search (using `hnswlib`)

**Key Python-Centric Requirements:**
- Benchmark scenarios for vector operations (e.g., magnitude calculation)
- Clean performance visualization using matplotlib
- HNSW index building/querying with scalability metrics
- Clear documentation of Python-C++ interop requirements
- Dependency management (numpy, hnswlib, pybind11)

**Proposed Workflow:**
```python
# Pure Python baseline
def magnitude_python(arr):
    return np.sqrt(np.sum(arr**2, axis=1))

# C++ accelerated version (via PyBind11)
import magnitude  # Pre-compiled module
magnitude.magnitude_cpp(data) 

# HNSW Integration
p = hnswlib.Index(space='l2', dim=128)
p.add_items(vectors)
neighbors, distances = p.knn_query(queries, k=10)


**Related Issue(s):**
Closes #ISSUE_ID

**Checklist:**
- [x] My code follows the style guidelines of this project
- [x] I have performed a self-review of my code
- [x] I have commented my code, especially in hard-to-understand areas
- [x] I have added necessary tests
- [x] All new and existing tests pass

**Screenshots (if applicable):**
<!-- Drag and drop or paste images here -->
