use std::hint::black_box;
use std::time::Instant;
use weight_space_microkernel::kernel::{AgentMicroKernel, FLAG_DANGER, OP_MSG_BASH, OP_MSG_SEARCH};

fn main() {
    let mut kernel = AgentMicroKernel::new();
    kernel.program_weights();
    println!(
        "Causal transformer: {} float parameters",
        kernel.parameter_count()
    );
    for tokens in [
        &[OP_MSG_SEARCH][..],
        &[OP_MSG_BASH][..],
        &[OP_MSG_BASH, FLAG_DANGER][..],
    ] {
        println!("{tokens:?} -> {:?}", kernel.forward(tokens).unwrap());
    }
    // Vary input, prevent elimination of the calls, and report actual elapsed time.
    const ITERATIONS: usize = 1_000;
    let start = Instant::now();
    for i in 0..ITERATIONS {
        let tokens = if i % 2 == 0 {
            &[OP_MSG_SEARCH][..]
        } else {
            &[OP_MSG_BASH, FLAG_DANGER][..]
        };
        black_box(kernel.forward(black_box(tokens)).unwrap());
    }
    let elapsed = start.elapsed();
    println!(
        "{ITERATIONS} mixed inputs in {elapsed:?}; {:.2} µs/inference",
        elapsed.as_secs_f64() * 1e6 / ITERATIONS as f64
    );
    println!(
        "Build: {}; architecture: {}; OS: {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        std::env::consts::ARCH,
        std::env::consts::OS
    );
}
