use std::time::Instant;
use weight_space_microkernel::kernel::{
    AgentMicroKernel, FLAG_DANGER, OP_MSG_BASH, OP_MSG_SEARCH, TOOL_BASH, TOOL_ORACLE,
    TOOL_SEARCH,
};

fn main() {
    println!("=== Weight-Space Programmed Agent Micro-Kernel (Rust) ===");
    println!("Model footprint: ~87,552 parameters (350 KB float32, L2/L3 cache resident)");

    let mut kernel = AgentMicroKernel::new();
    println!("1. Programming weights zero-shot (no backprop, no training)...");
    kernel.program_weights();
    println!("   ✓ Weights compiled.");

    println!("\n2. Testing OpenClaw Reactive Dispatch:");
    let res_search = kernel.forward(&[OP_MSG_SEARCH]);
    println!(
        "   - Event [OP_MSG_SEARCH]        -> Tool {} (Expected: {})",
        res_search, TOOL_SEARCH
    );
    assert_eq!(res_search, TOOL_SEARCH);

    let res_bash = kernel.forward(&[OP_MSG_BASH]);
    println!(
        "   - Event [OP_MSG_BASH]          -> Tool {} (Expected: {})",
        res_bash, TOOL_BASH
    );
    assert_eq!(res_bash, TOOL_BASH);

    let res_danger = kernel.forward(&[OP_MSG_BASH, FLAG_DANGER]);
    println!(
        "   - Event [BASH + FLAG_DANGER]   -> Tool {} (Expected: {} [Diverted to Oracle])",
        res_danger, TOOL_ORACLE
    );
    assert_eq!(res_danger, TOOL_ORACLE);

    println!("\n3. Testing Surgical Weight Editing (Micro-surgery):");
    println!("   Surgically lowering threshold bias on neuron 0 (Δb = -50.0)...");
    kernel.patch_threshold(0, -50.0);
    let res_patched = kernel.forward(&[OP_MSG_BASH, FLAG_DANGER]);
    println!(
        "   - Event [BASH + FLAG_DANGER] post-edit -> Tool {} (Divert gate successfully disarmed)",
        res_patched
    );
    assert_eq!(res_patched, TOOL_BASH);

    println!("\n4. Benchmark: Running 1,000,000 forward passes...");
    let iterations = 1_000_000;
    let start = Instant::now();
    let mut dummy_acc = 0;
    for _ in 0..iterations {
        dummy_acc += kernel.forward(&[OP_MSG_BASH]);
    }
    let duration = start.elapsed();
    let avg_ns = duration.as_nanos() as f64 / iterations as f64;
    let throughput = (iterations as f64 / duration.as_secs_f64()) as u64;

    println!("   ✓ Total time:      {:.2?}", duration);
    println!("   ✓ Latency per hop: {:.2} ns ({:.4} µs)", avg_ns, avg_ns / 1000.0);
    println!("   ✓ Throughput:      {} dispatches/sec", throughput);
    println!("   (Accumulator check: {})", dummy_acc);
    println!("\nAll systems operational.");
}
