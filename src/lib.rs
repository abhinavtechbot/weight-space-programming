//! # Weight-Space Programmed Agent Micro-Kernel
//!
//! A ~100k-parameter neural executive controller for reactive agents (e.g., OpenClaw).
//! Formulated for zero-shot weight synthesis and surgical in-place parameter edits
//! without backprop, training loops, or runtime dependencies.

pub mod kernel;

pub use kernel::AgentMicroKernel;
