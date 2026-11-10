//! Hand-programmed causal transformer for bounded routing experiments.
pub mod kernel;
pub mod text_circuit;
pub use kernel::{AgentMicroKernel, InputError, Route};
