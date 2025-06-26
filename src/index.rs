// src/index.rs

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::backend::AnnBackend;  //new added
use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;

/// A brute-force k-NN index with cached norms, Rayon parallelism,
/// and support for L1, L2, Cosine, Chebyshev, and Minkowski-p distances.
#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct AnnIndex {
    dim: usize,
    metric: Distance,
    minkowski_p: Option<f32>,
    entries: Vec<(i64, Vec<f32>, f32)>, // (id, vector, squared_norm)
}

#[pymethods]
impl AnnIndex {
    #[new]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        Ok(AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
        })
    }

    #[staticmethod]
    pub fn new_minkowski(dim: usize, p: f32) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Invalid Dimension", "Dimension must be > 0"));
        }
        if p <= 0.0 {
            return Err(RustAnnError::py_err("Minkowski Error", "`p` must be > 0 for Minkowski distance"));
        }
        Ok(AnnIndex {
            dim,
            metric: Distance::Euclidean, // placeholder
            minkowski_p: Some(p),
            entries: Vec::new(),
        })
    }

    pub fn add(
        &mut self,
        _py: Python,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let view = data.as_array();
        let ids = ids.as_slice()?;
        if view.nrows() != ids.len() {
            return Err(RustAnnError::py_err("Input Mismatch", "`data` and `ids` must have same length"));
        }

        for (row, &id) in view.outer_iter().zip(ids) {
            let v = row.to_vec();
            if v.len() != self.dim {
                return Err(RustAnnError::py_err(
                    "Dimension Error",
                    format!("Expected dimension {}, got {}", self.dim, v.len()),
                ));
            }
            let sq_norm = v.iter().map(|x| x * x).sum::<f32>();
            self.entries.push((id, v, sq_norm));
        }
        Ok(())
    }

    pub fn remove(&mut self, ids: Vec<i64>) -> PyResult<()> {
        if !ids.is_empty() {
            let to_rm: std::collections::HashSet<i64> = ids.into_iter().collect();
            self.entries.retain(|(id, _, _)| !to_rm.contains(id));
        }
        Ok(())
    }

    pub fn search(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_slice()?;
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();

        let result: PyResult<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            self.inner_search(q, q_sq, k)
        });
        let (ids, dists) = result?;

        Ok((
            ids.into_pyarray(py).to_object(py),
            dists.into_pyarray(py).to_object(py),
        ))
    }

    pub fn search_batch(
        &self,
        py: Python,
        data: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<(PyObject, PyObject)> {
        let arr = data.as_array();
        let n = arr.nrows();

        let results: Vec<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = arr.row(i);
                    let q: Vec<f32> = row.to_vec();
                    let q_sq = q.iter().map(|x| x * x).sum::<f32>();
                    self.inner_search(&q, q_sq, k).unwrap()
                })
                .collect::<Vec<_>>()
        });

        let mut all_ids = Vec::with_capacity(n * k);
        let mut all_dists = Vec::with_capacity(n * k);
        for (ids, dists) in results {
            all_ids.extend(ids);
            all_dists.extend(dists);
        }

        let ids_arr: Array2<i64> = Array2::from_shape_vec((n, k), all_ids)
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape ids failed: {}", e)))?;
        let dists_arr: Array2<f32> = Array2::from_shape_vec((n, k), all_dists)
            .map_err(|e| RustAnnError::py_err("Reshape Error", format!("Reshape dists failed: {}", e)))?;

        Ok((
            ids_arr.into_pyarray(py).to_object(py),
            dists_arr.into_pyarray(py).to_object(py),
        ))
    }

    /// Search using a Python filter callback (e.g., to exclude some ids).
    pub fn search_filter_py(
        &self,
        py: Python,
        query: PyReadonlyArray1<f32>,
        k: usize,
        filter_fn: PyObject, // Python function: fn(id) -> bool
    ) -> PyResult<(PyObject, PyObject)> {
        let q = query.as_slice()?;
        let q_sq = q.iter().map(|x| x * x).sum::<f32>();
        let mut filtered: Vec<(i64, f32)> = Vec::new();

        for (id, vec, vec_sq) in self.entries.iter() {
            let allow = filter_fn.call1(py, (*id,))?.extract::<bool>(py)?;
            if !allow {
                continue;
            }
            let dist = self.compute_distance(q, q_sq, vec, *vec_sq);
            filtered.push((*id, dist));
        }

        filtered.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        filtered.truncate(k);

        let ids: Vec<i64> = filtered.iter().map(|(id, _)| *id).collect();
        let dists: Vec<f32> = filtered.iter().map(|(_, d)| *d).collect();

        Ok((
            ids.into_pyarray(py).to_object(py),
            dists.into_pyarray(py).to_object(py),
        ))
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let full = format!("{}.bin", path);
        save_index(self, &full).map_err(|e| e.into_pyerr())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let full = format!("{}.bin", path);
        load_index(&full).map_err(|e| e.into_pyerr())
    }
}

impl AnnIndex {
    /// Used internally by both normal and filtered search
    fn inner_search(&self, q: &[f32], q_sq: f32, k: usize) -> PyResult<(Vec<i64>, Vec<f32>)> {
        if q.len() != self.dim {
            return Err(RustAnnError::py_err("Dimension Error", format!(
                "Expected dimension {}, got {}", self.dim, q.len()
            )));
        }

        let results: Vec<(i64, f32)> = self.entries
            .par_iter()
            .map(|(id, vec, vec_sq)| {
                let dist = self.compute_distance(q, q_sq, vec, *vec_sq);
                (*id, dist)
            })
            .collect();

        let mut sorted = results;
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.truncate(k);

        let ids = sorted.iter().map(|(i, _)| *i).collect();
        let dists = sorted.iter().map(|(_, d)| *d).collect();
        Ok((ids, dists))
        

        
    }
}
impl AnnBackend for AnnIndex {
    fn new(dim: usize, metric: Distance) -> Self {
        AnnIndex {
            dim,
            metric,
            minkowski_p: None,
            entries: Vec::new(),
        }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        let id = self.entries.len() as i64;
        let sq_norm = item.iter().map(|x| x * x).sum::<f32>();
        self.entries.push((id, item, sq_norm));
    }

    fn build(&mut self) {
        // No-op for brute-force index
    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        let query_sq = vector.iter().map(|x| x * x).sum::<f32>();

        let mut results: Vec<(usize, f32)> = self.entries
            .iter()
            .enumerate()
            .map(|(idx, (_id, vec, vec_sq))| {
                let dot = vec.iter().zip(vector.iter()).map(|(x, y)| x * y).sum::<f32>();

                let dist = if let Some(p) = self.minkowski_p {
                    vec.iter().zip(vector.iter())
                        .map(|(x, y)| (x - y).abs().powf(p))
                        .sum::<f32>()
                        .powf(1.0 / p)
                } else {
                    match self.metric {
                        Distance::Euclidean => ((vec_sq + query_sq - 2.0 * dot).max(0.0)).sqrt(),
                        Distance::Cosine => {
                            let denom = vec_sq.sqrt().max(1e-12) * query_sq.sqrt().max(1e-12);
                            (1.0 - (dot / denom)).max(0.0)
                        }
                        Distance::Manhattan => vec.iter().zip(vector.iter())
                            .map(|(x, y)| (x - y).abs())
                            .sum::<f32>(),
                        Distance::Chebyshev => vec.iter().zip(vector.iter())
                            .map(|(x, y)| (x - y).abs())
                            .fold(0.0, f32::max),
                    }
                };

                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results.into_iter().map(|(i, _)| i).collect()
    }

    fn save(&self, path: &str) {
        let _ = save_index(self, path);
    }

    fn load(path: &str) -> Self {
        load_index(path).unwrap()
    }

    /// Compute distance between a query and stored vector
    fn compute_distance(&self, q: &[f32], q_sq: f32, vec: &[f32], vec_sq: f32) -> f32 {
        let dot = vec.iter().zip(q.iter()).map(|(x, y)| x * y).sum::<f32>();
        if let Some(p) = self.minkowski_p {
            let sum_p = vec.iter().zip(q.iter())
                .map(|(x, y)| (x - y).abs().powf(p))
                .sum::<f32>();
            return sum_p.powf(1.0 / p);
        }

        match self.metric {
            Distance::Euclidean => ((vec_sq + q_sq - 2.0 * dot).max(0.0)).sqrt(),
            Distance::Cosine => {
                let denom = vec_sq.sqrt().max(1e-12) * q_sq.sqrt().max(1e-12);
                (1.0 - (dot / denom)).max(0.0)
            }
            Distance::Manhattan => vec.iter().zip(q.iter())
                .map(|(x, y)| (x - y).abs())
                .sum(),
            Distance::Chebyshev => vec.iter().zip(q.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max),
        }
    }
}
