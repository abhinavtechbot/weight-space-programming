"""
AgentMicroKernel: 100k-parameter weight-space programmed controller.
Directly programmed without gradient descent, backprop, or external dependencies.
"""
import math

class AgentMicroKernel:
    def __init__(self, d_model=64, d_mlp=128, vocab_size=128):
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.vocab_size = vocab_size

        # Weight tensors (~87k params)
        self.W_emb = [[0.0] * d_model for _ in range(vocab_size)]
        
        # Layer 0 Attention & MLP
        self.W_q0 = [[0.0] * d_model for _ in range(d_model)]
        self.W_k0 = [[0.0] * d_model for _ in range(d_model)]
        self.W_v0 = [[0.0] * d_model for _ in range(d_model)]
        self.W_o0 = [[0.0] * d_model for _ in range(d_model)]
        self.W_mlp1_0 = [[0.0] * d_mlp for _ in range(d_model)]
        self.b_mlp1_0 = [0.0] * d_mlp
        self.W_mlp2_0 = [[0.0] * d_model for _ in range(d_mlp)]
        self.b_mlp2_0 = [0.0] * d_model

        # Output projection head
        self.W_head = [[0.0] * vocab_size for _ in range(d_model)]

    def program_weights(self):
        """
        Synthesizes weights to implement OpenClaw routing logic:
        - OP_MSG_SEARCH (10) -> TOOL_SEARCH (50)
        - OP_MSG_BASH   (11) -> TOOL_BASH (51)
        - Safety Check: Dangerous command (FLAG_DANGER: 12) -> Divert to TOOL_ORACLE (52)
        """
        OP_MSG_SEARCH = 10
        OP_MSG_BASH   = 11
        FLAG_DANGER   = 12
        TOOL_SEARCH   = 50
        TOOL_BASH     = 51
        TOOL_ORACLE   = 52

        # 1. Embedding Projection (Subspace mapping)
        self.W_emb[OP_MSG_SEARCH][16] = 10.0  # Intent channel: Search
        self.W_emb[OP_MSG_BASH][17]   = 10.0  # Intent channel: Bash
        self.W_emb[FLAG_DANGER][32]   = 10.0  # Dangerous flag channel

        # 2. Layer 0 Attention (Key-Value Matching)
        self.W_q0[16][16] = 1.0; self.W_k0[16][16] = 1.0; self.W_v0[16][20] = 5.0
        self.W_q0[17][17] = 1.0; self.W_k0[17][17] = 1.0; self.W_v0[17][21] = 5.0
        self.W_o0[20][20] = 1.0
        self.W_o0[21][21] = 1.0

        # 3. Layer 0 MLP: Boolean AND Gate for Safety Override
        # Neuron 0 fires IF AND ONLY IF (Bash intent AND Danger flag active)
        # 1.0 * x[17] + 1.0 * x[32] - 15.0:
        # If only Bash: 10 + 0 - 15 = -5 <= 0 (Gate does not fire)
        # If Bash + Danger: 10 + 10 - 15 = +5 > 0 (Gate fires!)
        self.W_mlp1_0[17][0] = 1.0
        self.W_mlp1_0[32][0] = 1.0
        self.b_mlp1_0[0] = -15.0
        self.W_mlp2_0[0][22] = 20.0  # Boosts Oracle logit to divert execution

        # 4. Output Head: Project Action Channel to Tool Tokens
        self.W_head[20][TOOL_SEARCH] = 10.0
        self.W_head[21][TOOL_BASH]   = 10.0
        self.W_head[22][TOOL_ORACLE] = 20.0

    def forward(self, input_tokens):
        """Microsecond forward pass without GPU requirements or dependencies."""
        x = [0.0] * self.d_model
        for tok in input_tokens:
            for j in range(self.d_model):
                x[j] += self.W_emb[tok][j]

        # Self-Attention step
        q = [sum(x[i] * self.W_q0[i][j] for i in range(self.d_model)) for j in range(self.d_model)]
        k = [sum(x[i] * self.W_k0[i][j] for i in range(self.d_model)) for j in range(self.d_model)]
        v = [sum(x[i] * self.W_v0[i][j] for i in range(self.d_model)) for j in range(self.d_model)]

        attn = 1.0
        v_proj = [v[i] * attn for i in range(self.d_model)]
        o = [sum(v_proj[i] * self.W_o0[i][j] for i in range(self.d_model)) for j in range(self.d_model)]
        for j in range(self.d_model):
            x[j] += o[j]

        # MLP step
        h = [max(0.0, sum(x[i] * self.W_mlp1_0[i][j] for i in range(self.d_model)) + self.b_mlp1_0[j]) for j in range(self.d_mlp)]
        mlp_out = [sum(h[i] * self.W_mlp2_0[i][j] for i in range(self.d_mlp)) + self.b_mlp2_0[j] for j in range(self.d_model)]
        for j in range(self.d_model):
            x[j] += mlp_out[j]

        # Head projection
        logits = [sum(x[i] * self.W_head[i][j] for i in range(self.d_model)) for j in range(self.vocab_size)]
        return logits.index(max(logits))

if __name__ == "__main__":
    kernel = AgentMicroKernel()
    kernel.program_weights()
    
    # Test 1: Search event -> TOOL_SEARCH (50)
    tool = kernel.forward([10])
    print(f"Event OP_MSG_SEARCH (10) -> Routed Tool: {tool} (Expected: 50)")
    assert tool == 50

    # Test 2: Standard safe Bash event -> TOOL_BASH (51)
    tool = kernel.forward([11])
    print(f"Event OP_MSG_BASH   (11) -> Routed Tool: {tool} (Expected: 51)")
    assert tool == 51

    # Test 3: Dangerous Bash event -> Safety Gate diverts to TOOL_ORACLE (52)
    tool = kernel.forward([11, 12])
    print(f"Event BASH + DANGER (11, 12) -> Routed Tool: {tool} (Expected: 52)")
    assert tool == 52

    print("All weight-space execution tests passed successfully.")
