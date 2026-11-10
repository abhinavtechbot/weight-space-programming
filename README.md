# Weight-Space Programming: Routing Transformer

This repository is a standalone research prototype for directly programming a small transformer in weight space. It implements a bounded routing task without training or an external model: given a sequence of event tokens, the transformer returns Search, Bash, Oracle, or NoRoute.

The original routing circuit does **not** execute tools, integrate with OpenClaw, parse natural language, enforce a security policy, or implement an agent state machine. The separate bulb demo executes virtual bulb functions using a directly programmed text circuit. These circuits are experiments, not production guardrails. In particular, the original input danger flag is supplied by the caller, and its example edit deliberately disables the override.

## Architecture

The dependency-free Rust implementation is a two-layer, post-normalized causal transformer with a 64-dimensional residual stream, four 16-dimensional attention heads per layer, 128-dimensional ReLU feed-forward layers, learned-form token and positional embeddings, and a three-logit output head. Attention uses scaled dot products, a causal mask, and softmax. It stores **79,555 float32 parameters** (318,220 bytes for values, excluding container overhead). The head has three outputs rather than a 128-token vocabulary; the parameter count is obtained from the actual tensors.

`AgentMicroKernel::new()` constructs zero weights with unit normalization gains; `program_weights()` resets the model and directly writes a fixed routing circuit. Positional embeddings are present but set to zero by this particular program. The forward pass currently allocates intermediate vectors; no latency or cache-residency guarantee is claimed.

## Routing contract

`forward(&[usize]) -> Result<Route, InputError>` accepts between 1 and 63 event tokens, each less than 128. It appends a reserved query token at position 64 or earlier and reads the output there. The reserved token (127) cannot be supplied by callers. Empty, oversized, out-of-range, and reserved-token inputs return an error. Other unrecognized in-range tokens are inert.

Inputs are treated as a set for this program: order and repetition do not change the decision. Bash takes precedence over Search; Danger only matters when Bash is present.

| Present tokens | Result |
| --- | --- |
| Bash and Danger, with or without Search | Oracle |
| Bash without Danger | Bash |
| Search without Bash, with or without Danger | Search |
| Neither Search nor Bash | NoRoute |

The first layer uses uniform causal attention to collect event embeddings at the query position. Pairs of ReLU neurons turn the three event channels into saturated presence flags. A constant reference channel allows the second layer's ReLU gates to calculate the table above despite normalization. The final head reads the three gate outputs. All of this is fixed algebraic weight construction; there is no optimizer or LLM compiler.

`disable_danger_override()` edits two weights in the oracle gate. After the edit, Bash plus Danger yields NoRoute, while the standalone safe Bash and Search cases stay unchanged. This illustrates a targeted intervention and its tests; it is not evidence that arbitrary edits preserve unrelated behavior.

## Run

Install a current stable Rust toolchain, then:

```sh
cargo test
cargo run --release --bin weight-space-microkernel
```

For the live comparison, run `bash run-comparison.sh`. It builds the release text circuit and starts a server at `http://127.0.0.1:8765`. The script uses the Infisical CLI and dev project configured in `~/Developer/adk-course-dev` by default (override with `INFISICAL_CLI` and `INFISICAL_PROJECT_DIR`). It uses `INFISICAL_TOKEN` if set, otherwise looks for the `weight-space-infisical` entry in the macOS Keychain. If neither is present, sign in to Infisical when prompted. `OPENROUTER_API_KEY` remains on the server, and `OPENROUTER_MODEL` can override the default `google/gemini-2.5-flash`. Each LLM comparison makes a billable API request. Anyone with access to the server can trigger those requests.

The bulb demo compares three paths. `TextCircuit` is a small recurrent ReLU network whose embedding and recurrent weights were written directly by Codex; no gradients or pretrained language model are used. It receives ordered word tokens from a mechanical tokenizer and produces three independent bulb signals. Its vocabulary and exclusion rules are limited, so it does not provide general language understanding. The previous keyword matcher is retained as a baseline. An OpenRouter model receives the original text and three tool definitions, and its tool calls update a separate virtual panel. Inputs start with all bulbs off. The circuit's reported model time includes tokenization but excludes process startup; OpenRouter timing includes the API round trip. The included tests are development examples, not an independently held-out accuracy benchmark.

The executable prints representative decisions and times 1,000 alternating short inputs using `black_box`. The benchmark reports OS, architecture, build profile, elapsed time and average latency. Results are machine-specific and cover only those short inputs, not maximum-length inputs or a production workload.

The public API is in `src/kernel.rs`. Tests cover routing, ordering and repetition, maximum input length, invalid input, the edit, causal masking, positional embeddings, normalization, and parameter count.

## Research context and next steps

The broader idea is inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and [Tracr](https://arxiv.org/abs/2301.05062). This repository does not reproduce their experiments. Useful follow-up work includes comparing against a conventional rules engine, testing a larger range of programmed behaviors, and measuring accuracy and latency on representative workloads before considering integration with an agent framework.

The package metadata declares Apache-2.0; a license text has not yet been added to the repository.
