use hnsw_rs::prelude::*;
use crate::backend::AnnBackend;
use crate::metrics::Distance;

pub struct HnswIndex {
    index: Hnsw<'static, f32, DistL2>,
    dims: usize,
    user_ids: Vec<i64>, // Maps internal IDs to user IDs
}

impl AnnBackend for HnswIndex {
    fn new(dims: usize, _distance: Distance) -> Self {
        let index = Hnsw::new(
            16,     // M: number of bi-directional links
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
        let internal_id = self.user_ids.len();
        self.index.insert(&item, internal_id);
        // Placeholder for user ID - will be mapped in actual implementation
        self.user_ids.push(internal_id as i64);
    }

    fn build(&mut self) {
        // No-op for HNSW (built during insertion)
    }

    fn search(&self, vector: &[f32], k: usize) -> Vec<usize> {
        self.index
            .search(vector, k, 50) // ef = 50
            .iter()
            .map(|n| n.d_id)
            .collect()
    }

    fn save(&self, _path: &str) {
        unimplemented!("HNSW save not implemented yet");
    }

    fn load(_path: &str) -> Self {
        unimplemented!("HNSW load not implemented yet");
    }
}