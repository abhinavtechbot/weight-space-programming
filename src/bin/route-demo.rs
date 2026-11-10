use std::io::{self, Read};
use std::time::Instant;
use weight_space_microkernel::text_circuit::TextCircuit;

fn main() {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text).expect("read input");
    let model = TextCircuit::new();
    let start = Instant::now();
    let tokens = TextCircuit::tokens(&text);
    let result = model.forward(&tokens);
    let elapsed = start.elapsed().as_nanos();
    println!(
        "{}{}{} {}",
        u8::from(result[0]),
        u8::from(result[1]),
        u8::from(result[2]),
        elapsed
    );
    println!("{}", tokens.join(" | "));
}
