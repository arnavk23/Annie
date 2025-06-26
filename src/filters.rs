// src/filters.rs

use pyo3::prelude::*;

#[pyfunction]
pub fn search_filter(query: &str) -> Vec<String> {
    // Dummy logic (replace with real logic)
    vec![format!("type:{}", query), "year:2023".to_string()]
}

#[pyfunction]
pub fn search_possible_filter() -> Vec<String> {
    vec!["type".into(), "genre".into(), "year".into()]
}
