// src/index.rs

use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

use crate::storage::{save_index, load_index};
use crate::metrics::Distance;
use crate::errors::RustAnnError;

#[pyclass]
#[derive(Serialize, Deserialize)]
pub struct AnnIndex {
    dim: usize,
    metric: Distance,
    entries: Vec<(i64, Vec<f32>, f32)>,
}

#[pymethods]
impl AnnIndex {
    #[new]
    pub fn new(dim: usize, metric: Distance) -> PyResult<Self> {
        if dim == 0 {
            return Err(RustAnnError::py_err("Dimension must be > 0"));
        }
        Ok(AnnIndex { dim, metric, entries: Vec::new() })
    }

    pub fn add(
        &mut self,
        _py: Python,
        data: PyReadonlyArray2<f32>,
        ids: PyReadonlyArray1<i64>,
    ) -> PyResult<()> {
        let view = data.as_array();
        let ids_slice = ids.as_slice()?;
        if view.nrows() != ids_slice.len() {
            return Err(RustAnnError::py_err("`data` and `ids` must have same length"));
        }
        for (row, &id) in view.outer_iter().zip(ids_slice) {
            let vec = row.to_vec();
            if vec.len() != self.dim {
                return Err(RustAnnError::py_err(format!(
                    "Expected dimension {}, got {}", self.dim, vec.len()
                )));
            }
            let sq_norm = vec.iter().map(|x| x*x).sum::<f32>();
            self.entries.push((id, vec, sq_norm));
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
        let q_sq = q.iter().map(|x| x*x).sum::<f32>();

        // Release GIL during inner_search
        let result: PyResult<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            self.inner_search(q, q_sq, k)
        });
        let (ids, dists) = result?;  // handle PyResult here

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

        // Release GIL for the entire parallel batch
        let results: Vec<(Vec<i64>, Vec<f32>)> = py.allow_threads(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let row = arr.row(i);
                    let q: Vec<f32> = row.to_vec();
                    let q_sq = q.iter().map(|x| x*x).sum::<f32>();
                    // Safe to unwrap because dims are validated
                    self.inner_search(&q, q_sq, k).unwrap()
                })
                .collect::<Vec<_>>() // annotate collect
        });
        // No need to map_err here; into_pyarray happens under GIL

        // Flatten
        let mut all_ids = Vec::with_capacity(n * k);
        let mut all_dists = Vec::with_capacity(n * k);
        for (ids, dists) in results {
            all_ids.extend(ids);
            all_dists.extend(dists);
        }

        let ids_arr: Array2<i64> =
            Array2::from_shape_vec((n, k), all_ids)
                .map_err(|e| RustAnnError::py_err(format!("Reshape ids failed: {}", e)))?;
        let dists_arr: Array2<f32> =
            Array2::from_shape_vec((n, k), all_dists)
                .map_err(|e| RustAnnError::py_err(format!("Reshape dists failed: {}", e)))?;

        Ok((
            ids_arr.into_pyarray(py).to_object(py),
            dists_arr.into_pyarray(py).to_object(py),
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
    fn inner_search(&self, q: &[f32], q_sq: f32, k: usize) -> PyResult<(Vec<i64>, Vec<f32>)> {
        if q.len() != self.dim {
            return Err(RustAnnError::py_err(format!(
                "Expected dim {}, got {}", self.dim, q.len()
            )));
        }

        let mut v: Vec<(i64, f32)> = self.entries
            .par_iter()
            .map(|(id, vec, vec_sq)| {
                let dot = vec.iter().zip(q.iter()).map(|(x,y)| x*y).sum::<f32>();
                let dist = match self.metric {
                    Distance::Euclidean => ((vec_sq + q_sq - 2.0*dot).max(0.0)).sqrt(),
                    Distance::Cosine => {
                        let denom = vec_sq.sqrt().max(1e-12) * q_sq.sqrt().max(1e-12);
                        (1.0 - (dot / denom)).max(0.0)
                    }
                };
                (*id, dist)
            })
            .collect();

        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        v.truncate(k);

        let ids = v.iter().map(|(i, _)| *i).collect();
        let dists = v.iter().map(|(_, d)| *d).collect();
        Ok((ids, dists))
    }
}
