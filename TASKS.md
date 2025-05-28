

# ✅ Annie – Contributor Task Checklist (SSOC Edition)



## 📌 Contribution Categories

| Level     | Description                           | Points |
| --------- | ------------------------------------- | ------ |
| 🟢 Easy   | Good for first-time contributors      | 20     |
| 🟡 Medium | Requires some technical understanding | 30     |
| 🔴 Hard   | Involves core features or research    | 40     |

Here’s a proposed **Contributor Checklist** for Annie, with tasks grouped by difficulty and point-values. Feel free to pick any item, leave a comment on the issue, and submit a PR when you’re ready!

---

## 🟢 Easy (20 points)

* [ ] **Add usage example to README**
  Show how to build and query the brute-force `AnnIndex` in Python.
* [ ] **Document the concurrency module**
  Flesh out doc-comments in `src/concurrency.rs` and add a short how-to in the docs folder.
* [ ] **Clean up compiler warnings**
  Remove or justify all `unused_imports` and other warnings in `src/storage.rs`, `src/index.rs`.
* [ ] **Unit-test the `Distance` enum**
  Write small pytest tests to confirm L1/L2/Cosine/Manhattan/Chebyshev behaviors.
* [ ] **Add badges to README**
  CI status, PyPI version, license, and docs build.

## 🟠 Medium (30 points)

* [ ] **Implement Manhattan & Chebyshev in the brute-force index**
  Add `Distance::Manhattan` and `Distance::Chebyshev` to `src/metrics.rs` and update `inner_search`.
* [ ] **Add batch-benchmark CI job**
  Automate running `scripts/benchmark.py` on each PR and compare to baseline.
* [ ] **Improve error messages**
  Standardize all `RustAnnError::py_err(...)` strings in `src/errors.rs` to follow a template.
* [ ] **Publish to PyPI via GitHub Actions**
  Enhance the existing `CI.yml` to upload to TestPyPI on `workflow_dispatch`.
* [ ] **Example notebooks**
  Create a Jupyter notebook demonstrating speedup vs pure-Python and HNSW integration.

## 🔴 Hard (40 points)

* [ ] **Trait-based backend refactor**
  Define an `AnnBackend` trait, implement both brute-force and HNSW backends, and dispatch in a single `AnnIndex`.
* [ ] **Full HNSW-rs v0.3.1 integration**
  Replace brute-force with the enum/trait approach, expose `HnswIndex` alongside `AnnIndex`, remove optional flags.
* [ ] **Advanced filtering support**
  Expose `search_filter` and `search_possible_filter` to Python, plus tests and examples.
* [ ] **Cross-platform binary wheels**
  Extend `CI.yml` to build and upload Linux/Mac/Windows wheels automatically on tags.
* [ ] **Automated performance dashboards**
  Integrate with a service (e.g. GitHub Pages or Grafana) to track benchmark trends over time.

---

Feel free to comment “I’ll take this” on any task, then open a PR referencing the checklist item. Thank you for helping make Annie even better!
