use hnsw_rs::prelude::*;
use pyo3::prelude::*; // For PyResult
use pyo3::exceptions::PyNotImplementedError; // For PyErr::new
use crate::backend::AnnBackend;
use crate::metrics::Distance;

pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
    user_ids: Vec<i64>, // Maps internal ID → user ID
}

impl AnnBackend for HnswIndex {
    fn new(dims: usize, _distance: Distance) -> Self {
        let index = Hnsw::new(
            16,     // M
            10_000, // max elements
            16,     // ef_construction
            200,    // ef_search
            DistL2 {},
        );
        HnswIndex {
            index,
            dims,
            user_ids: Vec::new(),
        }
    }

    fn add_item(&mut self, item: Vec<f32>) {
        let internal_id = self.user_ids.len() as i64;
        self.insert(&item, internal_id); // Use internal ID as user ID
    }

    fn build(&mut self) {
        // No-op: HNSW builds during insertion
    }

   fn search(&self, vector: &[f32], k: usize) -> (Vec<i64>, Vec<f32>) {
        self.index
            .search(vector, k, 50)
            .iter()
            .map(|n| self.get_user_id(n.d_id)) // map to user ID
            .collect()
    }

    fn save(&self, _path: &str) -> PyResult<()> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "save() is not supported in hnsw-rs v0.3.2",
        ))
    }

    fn load(_path: &str) -> Self {
        unimplemented!("HNSW load not implemented yet");
    }
}

impl HnswIndex {
    pub fn insert(&mut self, item: &[f32], user_id: i64) {
        let internal_id = self.user_ids.len();
        self.index.insert((item, internal_id));
        self.user_ids.push(user_id);
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn get_user_id(&self, internal_id: usize) -> i64 {
        *self.user_ids.get(internal_id).unwrap_or(&-1)
    }
}
