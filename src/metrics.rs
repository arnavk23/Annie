// src/metrics.rs

use pyo3::prelude::*;
use serde::{Serialize, Deserialize};

/// Distance metric options for ANN search.
#[pyclass]
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub enum Distance {
    /// Euclidean (L2) distance
    Euclidean,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Manhattan (L1) distance
    Manhattan,
}

#[pymethods]
impl Distance {
    /// Class attribute for Euclidean distance
    #[classattr]
    pub const EUCLIDEAN: Distance = Distance::Euclidean;

    /// Class attribute for Cosine distance
    #[classattr]
    pub const COSINE: Distance = Distance::Cosine;

    /// Class attribute for Manhattan distance
    #[classattr]
    pub const MANHATTAN: Distance = Distance::Manhattan;

    /// String representation
    fn __repr__(&self) -> &'static str {
        match self {
            Distance::Euclidean => "Distance.EUCLIDEAN",
            Distance::Cosine    => "Distance.COSINE",
            Distance::Manhattan => "Distance.MANHATTAN",
        }
    }
}
