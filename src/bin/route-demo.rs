use std::time::Instant;
use weight_space_microkernel::kernel::ToolSelector;

fn main() {
    let mut tokens = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "left" => tokens.push(10),
            "middle" => tokens.push(11),
            "right" => tokens.push(12),
            "none" => tokens.push(1),
            _ => {
                eprintln!("Invalid token");
                std::process::exit(2);
            }
        }
    }
    let model = ToolSelector::new();
    let start = Instant::now();
    match model.forward(&tokens) {
        Ok(result) => println!(
            "{}{}{} {}",
            u8::from(result[0]),
            u8::from(result[1]),
            u8::from(result[2]),
            start.elapsed().as_nanos()
        ),
        Err(error) => {
            eprintln!("{error:?}");
            std::process::exit(2);
        }
    }
}
