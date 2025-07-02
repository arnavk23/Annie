pub mod index;
mod storage;
pub mod metrics;
mod errors;
mod concurrency;

mod backend;
pub mod hnsw_index;
mod index_enum;
mod filters; 

use pyo3::prelude::*;
use crate::backend::AnnBackend;
use crate::index::AnnIndex;
use crate::metrics::Distance;
use crate::concurrency::ThreadSafeAnnIndex;
use hnsw_index::HnswIndex;

#[pyclass]
pub struct PyHnswIndex {
    inner: HnswIndex,
}

#[pymethods]
impl PyHnswIndex {
    #[new]
    fn new(dims: usize) -> Self {
        PyHnswIndex {
            inner: HnswIndex::new(dims, Distance::Euclidean),
        }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        self.inner.add_item(item);
    }

    fn add(&mut self, py: Python, data: PyReadonlyArray2<f32>, ids: PyReadonlyArray1<i64>) -> PyResult<()> {
        let data_slice = data.as_slice()?;
        let ids_slice = ids.as_slice()?;
        
        for (i, vector) in data_slice.chunks_exact(self.inner.dims).enumerate() {
            self.inner.add_item(vector.to_vec());
        }
        Ok(())
    }

    fn build(&mut self) {
        self.inner.build();
    }

    fn search(&self, py: Python, query: PyReadonlyArray1<f32>, k: usize) -> PyResult<(PyObject, PyObject)> {
        let query_slice = query.as_slice()?;
        let internal_ids = self.inner.search(query_slice, k);
        
        // Convert to user IDs and get distances
        let user_ids = internal_ids.iter()
            .map(|&id| self.inner.user_ids[id] as i64)
            .collect::<Vec<_>>();
        
        // Placeholder distances
        let distances = vec![0.0; user_ids.len()];
        
        Ok((
            user_ids.into_pyarray(py).to_object(py),
            distances.into_pyarray(py).to_object(py),
        ))
    }

    fn save(&self, path: String) {
        self.inner.save(&path);
    }

    #[staticmethod]
    fn load(_path: String) -> Self {
        panic!("load() not supported in hnsw-rs v0.3.2");
    }
}

#[pymodule]
fn rust_annie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnIndex>()?;
    m.add_class::<Distance>()?;
    m.add_class::<ThreadSafeAnnIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    Ok(())
}

