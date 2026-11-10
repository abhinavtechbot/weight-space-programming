use std::time::Instant;
use weight_space_microkernel::kernel::{AgentMicroKernel, FLAG_DANGER, OP_MSG_BASH, OP_MSG_SEARCH};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut kernel = AgentMicroKernel::new();
    kernel.program_weights();
    let mut tokens = Vec::new();
    for arg in args {
        match arg.as_str() {
            "search" => tokens.push(OP_MSG_SEARCH),
            "bash" => tokens.push(OP_MSG_BASH),
            "danger" => tokens.push(FLAG_DANGER),
            "other" => tokens.push(1),
            _ => {
                eprintln!("Invalid event");
                std::process::exit(2);
            }
        }
    }
    let start = Instant::now();
    match kernel.forward(&tokens) {
        Ok(route) => println!("{:?} {}", route, start.elapsed().as_nanos()),
        Err(err) => {
            eprintln!("{err:?}");
            std::process::exit(2);
        }
    }
}
