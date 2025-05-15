// src/errors.rs

use pyo3::exceptions::{PyException, PyIOError};
use pyo3::PyErr;

/// A simple error type for the ANN library, used to convert Rust errors into Python exceptions.
#[derive(Debug)]
pub struct RustAnnError(pub String);

impl RustAnnError {
    /// Create a generic Python exception (`Exception`) with the given message.
    pub fn py_err(msg: impl Into<String>) -> PyErr {
        PyException::new_err(msg.into())
    }

    /// Create a RustAnnError wrapping an I/O error message.
    /// This is used internally in save/load to signal I/O or serialization failures.
    pub fn io_err(msg: impl Into<String>) -> RustAnnError {
        RustAnnError(msg.into())
    }

    /// Convert this RustAnnError into a Python `IOError` (`OSError`) exception.
    pub fn into_pyerr(self) -> PyErr {
        PyIOError::new_err(self.0)
    }
}
