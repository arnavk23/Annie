// src/metrics.rs

use pyo3::prelude::*;
use serde::{Serialize, Deserialize};

/// Distance metric options for ANN search.
#[pyclass]
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Distance {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

#[pymethods]
impl Distance {
    /// Class attribute for Euclidean distance
    #[classattr]
    pub const EUCLIDEAN: Distance = Distance::Euclidean;

    /// Class attribute for Cosine distance
    #[classattr]
    pub const COSINE: Distance = Distance::Cosine;

    /// String representation
    fn __repr__(&self) -> &'static str {
        match self {
            Distance::Euclidean => "Distance.Euclidean",
            Distance::Cosine    => "Distance.Cosine",
        }
    }
}
