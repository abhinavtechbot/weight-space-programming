# Weight-Space Programming for Micro-Agent Architectures

> Direct weight-space synthesis and surgical parameter editing of ~100k parameter neural micro-kernels for autonomous, local-first agent frameworks like [OpenClaw](https://github.com/openclaw/openclaw).

---

## 1. Executive Summary

Autonomous agent architectures (such as OpenClaw, AutoGen, and CrewAI) operate over proactive, event-driven loops: receiving incoming messages (Telegram, WhatsApp, Slack, webhooks), routing intents, extracting argument slots, validating policy guardrails, invoking external tools, and managing multi-step state machines.

Today, agent frameworks invoke giant frontier Large Language Models (70B–405B+ parameters) for **every single hop** of this control loop. This causes:
* **High Latency:** 500ms to 3,000ms per micro-decision.
* **Prohibitive Costs:** Continuous token drain for mechanical status checks.
* **Non-Determinism:** Susceptibility to formatting errors, jailbreaks, and state-machine derailment.

This repository formulates a **Weight-Space Programmed Micro-Kernel**: a small **~100k parameter neural model** (such as a 2-layer causal transformer or 1-layer looped universal transformer) whose weights are **directly written, inspected, and surgically edited by frontier LLMs without gradient descent or training loops**, as established in recent MIT CSAIL research ([Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/)).

```
+-------------------------------------------------------------------------+
|                  Frontier LLM (Weight-Space Compiler)                   |
|   Reads specs / tools -> Writes & patches 100k model weights (0-shot)   |
+-------------------------------------------------------------------------+
                                    │ (Direct Tensor Edits / Synthesis)
                                    ▼
+─────────────────────────────────────────────────────────────────────────+
|               100k Parameter Agent Micro-Kernel (Local)                 |
|   • Sub-millisecond inference (CPU cache / C++ / WebAssembly / pure NumPy) |
|   • Reactive Event Loop (Telegram, WhatsApp, Webhooks, Timers)          |
|   • Tool Dispatcher & Slot-Binder                                       |
|   • Finite State Machine (IDLE, AWAIT_TOOL, CHECK_EXIT, LOOP, HALT)     |
+─────────────────────────────────────────────────────────────────────────+
         │                                               │
         ▼ (Low-latency Execution)                       ▼ (Complex Synthesis)
  [Local Tools: bash, fs, cron]             [Oracle Frontier LLM (as a Tool)]
```

---

## 2. Research Foundations

### 2.1 Sparks of Weight-Space Programming (Takehi et al., 2026)
In *"Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"* (MIT CSAIL), Takehi, Teoh, Zhang, and Khattab showed that frontier LLMs possess an intrinsic understanding of raw weight matrices:
1. **Reading Weights (Interpretability):** An LLM can inspect raw weight tensors and identify what game a policy was trained to play (Breakout, Pong, Freeway) or determine whether a model was trained to be sycophantic without tool execution.
2. **Zero-Shot Weight Synthesis:** Without environment feedback or gradient updates, an LLM generated complete weight tensors for MinAtar games (Breakout, Seaquest, Asterix) that outperformed policies trained with PPO for 100M–500M steps. For a 1.48M parameter Go policy, an LLM wrote a 336-line Python synthesis script assembling layer weights that beat trained PPO baselines.
3. **Surgical Weight Editing (Micro-Surgery):** The LLM diagnosed a known reasoning loop in `Qwen3.5-0.8B`, surgically modifying only **0.016% of weights** (131,072 parameters), boosting MATH-500 accuracy from 7.9% to 48.1% without retraining or catastrophic forgetting.

### 2.2 RASP and Compiled Transformer Circuits (Weiss et al., 2021; Lindner et al., 2023)
Tracr (Google DeepMind) proved that symbolic algorithms expressed in RASP (Restricted Access Sequence Processing) can be compiled directly into transformer weights:
* **Attention Heads as Selectors:** Implementing bilinear key-value lookups and sequence routing.
* **MLPs as Element-wise Logic Gates:** Implementing threshold detectors and non-linear transitions.
* **Orthogonal Subspaces (Craft):** Using orthogonal slices of the residual stream as isolated variables.

### 2.3 OpenClaw Autonomous Agent Framework
OpenClaw provides a local-first gateway process executing proactive event loops over messaging interfaces and local tooling. Rather than querying a frontier LLM on every reactive tick, OpenClaw's core orchestration can be handed off to an ultra-fast, local 100k parameter neural controller.

---

## 3. Architecture Specification: The 100k "Nano-Executive"

The model is configured as a **2-Layer Causal Transformer** (or 1-Layer Universal Transformer looped dynamically) operating over discrete agent opcodes and ASCII chunks.

### 3.1 Parameter Allocation Budget

| Component | Dimensions / Shape | Param Count |
| :--- | :--- | :--- |
| **Opcode & Byte Vocabulary** | $V = 128$ tokens $\times d_{\text{model}} = 64$ | 8,192 |
| **Positional Embeddings** | Max sequence length $L = 64 \times d_{\text{model}} = 64$ | 4,096 |
| **Layer 0 Self-Attention** | 4 heads ($d_k = 16$), $W_Q, W_K, W_V, W_O \in \mathbb{R}^{64 \times 64}$ + biases | 16,640 |
| **Layer 0 MLP Block** | Linear $64 \to 128$ (ReLU) + Linear $128 \to 64$ + biases | 16,576 |
| **Layer 0 Normalization** | 2 $\times$ LayerNorm ($d_{\text{model}} = 64$) | 256 |
| **Layer 1 Self-Attention** | 4 heads ($d_k = 16$), $W_Q, W_K, W_V, W_O \in \mathbb{R}^{64 \times 64}$ + biases | 16,640 |
| **Layer 1 MLP Block** | Linear $64 \to 128$ (ReLU) + Linear $128 \to 64$ + biases | 16,576 |
| **Layer 1 Normalization** | 2 $\times$ LayerNorm ($d_{\text{model}} = 64$) | 256 |
| **Output Head & Final Norm** | Final LayerNorm ($128$) + Un-tied Head ($64 \times 128$) | 8,320 |
| **Total Parameter Count** | | **87,552** |

*(For a full 256-byte ASCII/UTF-8 token table, the total parameter count is **108,032**, sitting right at the ~100k parameter target).*

### 3.2 Residual Stream Subspace Allocation

The 64 dimensions of the hidden state vector $x \in \mathbb{R}^{64}$ are partitioned into dedicated functional registers:

```
[0 ................. 15]  State Machine Register (IDLE, PENDING, TOOL_WAIT, RETRY, HALT)
[16 ................ 31]  Action & Tool Dispatch Channels (BASH, FS_READ, WEB, SEND_MSG, ORACLE)
[32 ................ 47]  Policy Guardrails & Security Flags (ALLOW, REQUIRE_CONFIRM, READ_ONLY)
[48 ................ 63]  Pointer Slots & Counters (Step counter, exit codes, retry budget)
```

---

## 4. How the Weights are Programmed

Rather than computing backward passes with SGD:
1. **Frontier LLM as the Meta-Compiler:** Given a high-level agent specification (e.g., tools, triggers, permissions), the frontier LLM emits a standalone Python script that calculates and writes the tensor weights directly.
2. **Attention Matrices as Exact Routers:**
   $$A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
   $W_Q$ and $W_K$ project specific intent subspace dimensions to create unit dot products, binding incoming trigger events to tool action channels.
3. **MLP Neurons as Threshold Boolean Gates:**
   $$h = \text{ReLU}(W_{\text{mlp1}} x + b_{\text{mlp1}})$$
   Weights and negative biases implement strict logical thresholds:
   $$\text{ReLU}(w_1 \cdot \text{intent}_{\text{bash}} + w_2 \cdot \text{flag}_{\text{dangerous}} - \theta)$$
4. **Surgical In-Place Edits (Micro-Surgery):**
   * **Adding a Tool:** The frontier LLM leaves existing weights intact and programs an unused dimension slice (e.g. index 23) in $W_Q, W_K, W_{\text{head}}$.
   * **Fixing Infinite Loops / Retries:** The frontier LLM alters single scalar bias terms on the threshold neuron ($\Delta b = -2.5$), terminating loops without retraining or catastrophic forgetting.

---

## 5. Reference Implementation (Rust)

The repository provides a zero-dependency **Rust Crate** (`src/`) engineered for production-grade, sub-microsecond edge execution:
* **Zero Allocation During Forward Passes:** The entire residual stream ($x \in \mathbb{R}^{64}$) and intermediate projections live on the stack or in flat contiguous memory.
* **L2/L3 CPU Cache Residency:** At ~87k parameters in float32 (~350 KB) or FP16 (~175 KB), the entire model remains continuously resident in hardware cache.
* **Sub-Microsecond Latency:** Target latency is **< 1 µs per dispatch hop** (> 1,000,000 dispatches/sec on modern CPUs), compared to 500ms–3s for cloud LLMs.
* **WASM / Embedded Ready:** Can be built as a standalone binary or compiled to `wasm32-unknown-unknown` for sandboxed execution in browsers, edge workers, or daemon plugins.

### Building and Testing

```bash
# Run tests (verifying routing, Boolean threshold gates, and surgical patching)
cargo test

# Run high-throughput release benchmark
cargo run --release
```

### Core Implementation Layout

* **[`src/kernel.rs`](src/kernel.rs)**: Defines `AgentMicroKernel`, matrix dimensions, zero-shot weight synthesis (`program_weights`), surgical parameter editing (`patch_threshold`), and the inline forward execution pass.
* **[`src/main.rs`](src/main.rs)**: Standalone benchmark and execution suite testing intent routing, safety overrides, and high-throughput dispatch loops.

---

## 6. Comparison: Traditional Agents vs. Weight-Space Agents

| Metric / Dimension | Traditional Agent Loop (e.g. Pure LLM) | Weight-Space Programmed Micro-Kernel |
| :--- | :--- | :--- |
| **Model Size in Loop** | 70B – 405B parameters | **~100,000 parameters** |
| **Decision Latency** | 500ms – 3,000ms per step | **< 1ms (sub-millisecond)** |
| **Memory Footprint** | 16GB – 80GB VRAM | **< 1MB (runs in L2/L3 CPU cache)** |
| **Inference Cost** | Constant API token expenditure | **Zero cost for routine loop iterations** |
| **Modifying Behaviors** | Complex prompt engineering / LoRA fine-tuning | **Direct in-place weight edits via compiler script** |
| **Failure Modes** | Hallucinations, syntax errors, loop traps | **Deterministic finite state machine guarantees** |

---

## 7. References and Citations

1. **Takehi, R., Teoh, J., Zhang, A., & Khattab, O.** (2026). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*. MIT CSAIL. [https://rikiyat.github.io/blog/weight-space-programming/](https://rikiyat.github.io/blog/weight-space-programming/)
2. **Weiss, G., Goldberg, Y., & Yahav, E.** (2021). *Thinking Like Transformers*. International Conference on Machine Learning (ICML).
3. **Lindner, D., Kramár, J., Farquhar, S., Rahtz, M., McGrath, T., & Mikulik, V.** (2023). *Tracr: Compiled Transformers as a Laboratory for Interpretability*. arXiv:2301.05062.
4. **OpenClaw Community.** (2026). *OpenClaw: Autonomous, Local-First Personal AI Assistant*. [https://github.com/openclaw/openclaw](https://github.com/openclaw/openclaw)
5. **Elhage, N., Nanda, N., Olsson, C., et al.** (2021). *A Mathematical Framework for Transformer Circuits*. Transformer Circuits Thread.

### BibTeX

```bibtex
@article{takehi2026weightspace,
  title   = {{Sparks of Weight-Space Programming: LLMs can natively read \& write neural network weights}},
  author  = {Takehi, Rikiya and Teoh, Jayden and Zhang, Alex and Khattab, Omar},
  journal = {MIT CSAIL Blog},
  year    = {2026},
  url     = {https://rikiyat.github.io/blog/weight-space-programming/}
}

@inproceedings{lindner2023tracr,
  title     = {{Tracr: Compiled Transformers as a Laboratory for Interpretability}},
  author    = {David Lindner and J{\'a}nos Kram{\'a}r and Sebastian Farquhar and Matthew Rahtz and Thomas McGrath and Vladimir Mikulik},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@article{weiss2021thinking,
  title   = {{Thinking Like Transformers}},
  author  = {Gail Weiss and Yoav Goldberg and Eran Yahav},
  journal = {International Conference on Machine Learning (ICML)},
  year    = {2021}
}
```

---

## 8. License

This repository is licensed under the Apache 2.0 License.
