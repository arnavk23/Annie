mod index;
mod storage;
pub mod metrics;
mod filters;
mod errors;
mod concurrency;

use pyo3::prelude::*;
use index::AnnIndex;
use metrics::Distance;
use concurrency::ThreadSafeAnnIndex;
use filters::{search_filter, search_possible_filter}; // Import the new functions

/// The Python module declaration.
#[pymodule]
fn rust_annie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AnnIndex>()?;
    m.add_class::<Distance>()?;
    m.add_class::<ThreadSafeAnnIndex>()?;
    m.add_function(wrap_pyfunction!(search_filter, m)?)?;
    m.add_function(wrap_pyfunction!(search_possible_filter, m)?)?;
    Ok(())
}
