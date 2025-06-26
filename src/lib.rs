
mod index;
mod storage;
pub mod metrics;
mod filters;
mod errors;
mod concurrency;

mod backend;
mod hnsw_index;
mod index_enum;

use pyo3::prelude::*;

use index::AnnIndex;
use metrics::Distance;
use concurrency::ThreadSafeAnnIndex;
use filters::{search_filter, search_possible_filter};

use crate::backend::AnnBackend;
use crate::index::AnnIndex;
use crate::metrics::Distance;
use crate::concurrency::ThreadSafeAnnIndex;
use crate::hnsw_index::HnswIndex;

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

    fn build(&mut self) {
        self.inner.build();
    }

    fn search(&self, vector: Vec<f32>, k: usize) -> Vec<usize> {
        self.inner.search(&vector, k)
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

    m.add_function(wrap_pyfunction!(search_filter, m)?)?;
    m.add_function(wrap_pyfunction!(search_possible_filter, m)?)?;
    m.add_class::<PyHnswIndex>()?;
    Ok(())
}

