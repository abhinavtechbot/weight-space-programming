//! A small causal transformer and a hand-constructed routing program.

pub const D_MODEL: usize = 64;
pub const D_MLP: usize = 128;
pub const VOCAB_SIZE: usize = 128;
pub const MAX_SEQUENCE: usize = 64;
pub const HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / HEADS;
const EPS: f32 = 1e-5;

pub const OP_MSG_SEARCH: usize = 10;
pub const OP_MSG_BASH: usize = 11;
pub const FLAG_DANGER: usize = 12;
/// Reserved internally as the last position, where the routing result is read.
pub const OP_QUERY: usize = 127;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Search,
    Bash,
    Oracle,
    NoRoute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    Empty,
    TooLong { max: usize },
    InvalidToken { index: usize, token: usize },
    ReservedToken { index: usize },
}

#[derive(Clone)]
struct Linear {
    weights: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}
impl Linear {
    fn new(input: usize, output: usize) -> Self {
        Self {
            weights: vec![0.0; input * output],
            bias: vec![0.0; output],
            input,
            output,
        }
    }
    fn set(&mut self, i: usize, j: usize, value: f32) {
        self.weights[i * self.output + j] = value;
    }
    fn apply(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input);
        let mut out = self.bias.clone();
        for (i, &v) in input.iter().enumerate() {
            if v != 0.0 {
                for (j, x) in out.iter_mut().enumerate() {
                    *x += v * self.weights[i * self.output + j];
                }
            }
        }
        out
    }
    fn count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

#[derive(Clone)]
struct Norm {
    gain: Vec<f32>,
    bias: Vec<f32>,
}
impl Norm {
    fn new() -> Self {
        Self {
            gain: vec![1.0; D_MODEL],
            bias: vec![0.0; D_MODEL],
        }
    }
    fn apply(&self, x: &mut [f32]) {
        let mean = x.iter().sum::<f32>() / D_MODEL as f32;
        let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / D_MODEL as f32;
        let scale = 1.0 / (variance + EPS).sqrt();
        for (i, v) in x.iter_mut().enumerate() {
            *v = (*v - mean) * scale * self.gain[i] + self.bias[i];
        }
    }
    fn count(&self) -> usize {
        self.gain.len() + self.bias.len()
    }
}

#[derive(Clone)]
struct Block {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    up: Linear,
    down: Linear,
    norm1: Norm,
    norm2: Norm,
}
impl Block {
    fn new() -> Self {
        Self {
            q: Linear::new(D_MODEL, D_MODEL),
            k: Linear::new(D_MODEL, D_MODEL),
            v: Linear::new(D_MODEL, D_MODEL),
            o: Linear::new(D_MODEL, D_MODEL),
            up: Linear::new(D_MODEL, D_MLP),
            down: Linear::new(D_MLP, D_MODEL),
            norm1: Norm::new(),
            norm2: Norm::new(),
        }
    }
    fn count(&self) -> usize {
        self.q.count()
            + self.k.count()
            + self.v.count()
            + self.o.count()
            + self.up.count()
            + self.down.count()
            + self.norm1.count()
            + self.norm2.count()
    }
    fn forward(&self, x: &mut [Vec<f32>]) {
        let queries: Vec<_> = x.iter().map(|v| self.q.apply(v)).collect();
        let keys: Vec<_> = x.iter().map(|v| self.k.apply(v)).collect();
        let values: Vec<_> = x.iter().map(|v| self.v.apply(v)).collect();
        for t in 0..x.len() {
            let mut context = vec![0.0; D_MODEL];
            for head in 0..HEADS {
                let start = head * HEAD_DIM;
                let mut scores: Vec<f32> = (0..=t)
                    .map(|s| {
                        (start..start + HEAD_DIM)
                            .map(|j| queries[t][j] * keys[s][j])
                            .sum::<f32>()
                            / (HEAD_DIM as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for score in &mut scores {
                    *score = (*score - max).exp();
                }
                let total: f32 = scores.iter().sum();
                for (s, score) in scores.iter().enumerate() {
                    for j in start..start + HEAD_DIM {
                        context[j] += score / total * values[s][j];
                    }
                }
            }
            let attention = self.o.apply(&context);
            for (a, b) in x[t].iter_mut().zip(attention) {
                *a += b;
            }
            self.norm1.apply(&mut x[t]);
            let mut hidden = self.up.apply(&x[t]);
            for v in &mut hidden {
                *v = v.max(0.0);
            }
            let update = self.down.apply(&hidden);
            for (a, b) in x[t].iter_mut().zip(update) {
                *a += b;
            }
            self.norm2.apply(&mut x[t]);
        }
    }
}

/// Two-layer, post-normalized causal transformer with a separately programmed circuit.
#[derive(Clone)]
pub struct AgentMicroKernel {
    embeddings: Vec<f32>,
    positions: Vec<f32>,
    blocks: [Block; 2],
    final_norm: Norm,
    head: Linear,
}
impl Default for AgentMicroKernel {
    fn default() -> Self {
        Self::new()
    }
}
impl AgentMicroKernel {
    /// Constructs zero weights (normalization gains are initialized to one).
    pub fn new() -> Self {
        Self {
            embeddings: vec![0.0; VOCAB_SIZE * D_MODEL],
            positions: vec![0.0; MAX_SEQUENCE * D_MODEL],
            blocks: [Block::new(), Block::new()],
            final_norm: Norm::new(),
            head: Linear::new(D_MODEL, 3),
        }
    }
    /// Number of stored float parameters, including biases and normalization parameters.
    pub fn parameter_count(&self) -> usize {
        self.embeddings.len()
            + self.positions.len()
            + self.blocks.iter().map(Block::count).sum::<usize>()
            + self.final_norm.count()
            + self.head.count()
    }
    /// Reset and directly synthesize the bounded routing circuit, without training.
    pub fn program_weights(&mut self) {
        *self = Self::new();
        for (token, channel) in [(OP_MSG_SEARCH, 0), (OP_MSG_BASH, 1), (FLAG_DANGER, 2)] {
            self.embeddings[token * D_MODEL + channel] = 1.0;
        }
        let first = &mut self.blocks[0];
        // Zero queries and keys produce uniform causal attention. Copy event channels.
        for i in 0..3 {
            first.v.set(i, i, 1.0);
            first.o.set(i, i, 1.0);
        }
        // At the query position, x[i]-x[3] is positive iff that token occurred.
        // Difference of two ReLUs saturates each presence flag at one.
        for i in 0..3 {
            first.up.set(i, 2 * i, 1.0);
            first.up.set(3, 2 * i, -1.0);
            first.up.set(i, 2 * i + 1, 1.0);
            first.up.set(3, 2 * i + 1, -1.0);
            first.up.bias[2 * i + 1] = -0.01;
            first.down.set(2 * i, 8 + i, 100.0);
            first.down.set(2 * i + 1, 8 + i, -100.0);
        }
        // An always-on reference lets later gates compare normalized flags.
        first.down.bias[12] = 1.0;
        let second = &mut self.blocks[1];
        // Search iff search and not bash; bash iff bash and not danger;
        // oracle iff bash and danger. Channels 8..10 are presence flags.
        for (channel, value) in [(8, 1.0), (9, -1.0)] {
            second.up.set(channel, 0, value);
        }
        for (channel, value) in [(9, 1.0), (10, -1.0)] {
            second.up.set(channel, 1, value);
        }
        for (channel, value) in [(9, 1.0), (10, 1.0), (12, -1.0), (11, -1.0)] {
            second.up.set(channel, 2, value);
        }
        // Each expression has zero coefficient sum, so normalization offsets cancel.
        for i in 0..3 {
            second.down.set(i, 16 + i, 100.0);
            self.head.set(16 + i, i, 1.0);
            self.head.set(19, i, -1.0);
        }
    }
    /// Disable the experimental danger override by raising its reference threshold.
    /// A dangerous bash input then produces NoRoute; safe routes are unaffected.
    pub fn disable_danger_override(&mut self) {
        self.blocks[1].up.set(12, 2, -3.0);
        self.blocks[1].up.set(11, 2, 1.0);
    }
    /// Evaluate 1..=63 event tokens, followed internally by a reserved query token.
    pub fn forward(&self, tokens: &[usize]) -> Result<Route, InputError> {
        if tokens.is_empty() {
            return Err(InputError::Empty);
        }
        if tokens.len() >= MAX_SEQUENCE {
            return Err(InputError::TooLong {
                max: MAX_SEQUENCE - 1,
            });
        }
        for (index, &token) in tokens.iter().enumerate() {
            if token >= VOCAB_SIZE {
                return Err(InputError::InvalidToken { index, token });
            }
            if token == OP_QUERY {
                return Err(InputError::ReservedToken { index });
            }
        }
        let mut x: Vec<Vec<f32>> = tokens
            .iter()
            .copied()
            .chain(std::iter::once(OP_QUERY))
            .enumerate()
            .map(|(pos, tok)| {
                (0..D_MODEL)
                    .map(|j| self.embeddings[tok * D_MODEL + j] + self.positions[pos * D_MODEL + j])
                    .collect()
            })
            .collect();
        for block in &self.blocks {
            block.forward(&mut x);
        }
        let last = x.last_mut().expect("query token is present");
        self.final_norm.apply(last);
        let logits = self.head.apply(last);
        let (index, &value) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        // Roundoff near zero must not turn an absent route into a decision.
        if value <= 1e-5 {
            Ok(Route::NoRoute)
        } else {
            Ok([Route::Search, Route::Bash, Route::Oracle][index])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn programmed() -> AgentMicroKernel {
        let mut k = AgentMicroKernel::new();
        k.program_weights();
        k
    }
    #[test]
    fn dimensions_and_count() {
        assert_eq!(D_MODEL % HEADS, 0);
        assert_eq!(programmed().parameter_count(), 79_555);
    }
    #[test]
    fn routing_and_precedence() {
        let k = programmed();
        for (input, expected) in [
            (vec![OP_MSG_SEARCH], Route::Search),
            (vec![OP_MSG_BASH], Route::Bash),
            (vec![OP_MSG_BASH, FLAG_DANGER], Route::Oracle),
            (vec![FLAG_DANGER, OP_MSG_BASH], Route::Oracle),
            (vec![OP_MSG_SEARCH, OP_MSG_BASH], Route::Bash),
            (vec![OP_MSG_SEARCH, FLAG_DANGER], Route::Search),
            (vec![FLAG_DANGER], Route::NoRoute),
            (vec![1], Route::NoRoute),
            (
                vec![OP_MSG_BASH, OP_MSG_SEARCH, FLAG_DANGER, OP_MSG_BASH],
                Route::Oracle,
            ),
        ] {
            assert_eq!(k.forward(&input), Ok(expected), "{input:?}");
        }
    }
    #[test]
    fn max_length_and_repetition() {
        let k = programmed();
        let mut input = vec![1; 63];
        input[0] = OP_MSG_BASH;
        input[62] = FLAG_DANGER;
        assert_eq!(k.forward(&input), Ok(Route::Oracle));
        assert_eq!(k.forward(&vec![OP_MSG_SEARCH; 63]), Ok(Route::Search));
    }
    #[test]
    fn invalid_inputs() {
        let k = programmed();
        assert_eq!(k.forward(&[]), Err(InputError::Empty));
        assert_eq!(
            k.forward(&vec![1; 64]),
            Err(InputError::TooLong { max: 63 })
        );
        assert_eq!(
            k.forward(&[128]),
            Err(InputError::InvalidToken {
                index: 0,
                token: 128
            })
        );
        assert_eq!(
            k.forward(&[OP_QUERY]),
            Err(InputError::ReservedToken { index: 0 })
        );
    }
    #[test]
    fn edit_is_local_to_override() {
        let mut k = programmed();
        k.disable_danger_override();
        assert_eq!(k.forward(&[OP_MSG_BASH, FLAG_DANGER]), Ok(Route::NoRoute));
        assert_eq!(k.forward(&[OP_MSG_BASH]), Ok(Route::Bash));
        assert_eq!(k.forward(&[OP_MSG_SEARCH]), Ok(Route::Search));
    }
    #[test]
    fn causal_attention_cannot_see_future() {
        let mut block = Block::new();
        block.v.set(0, 0, 1.0);
        block.o.set(0, 0, 1.0);
        let mut a = vec![vec![0.0; D_MODEL], vec![0.0; D_MODEL]];
        let mut b = a.clone();
        b[1][0] = 9.0;
        block.forward(&mut a);
        block.forward(&mut b);
        assert_eq!(a[0], b[0]);
    }
    #[test]
    fn normalization_is_finite_and_centered() {
        let norm = Norm::new();
        let mut x = vec![0.0; D_MODEL];
        x[0] = 1.0;
        norm.apply(&mut x);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(x.iter().sum::<f32>().abs() < 1e-4);
    }
    #[test]
    fn position_embedding_is_used() {
        let mut k = programmed();
        let baseline = k.forward(&[OP_MSG_SEARCH]).unwrap();
        k.positions[D_MODEL + 1] = 100.0;
        assert_ne!(k.forward(&[OP_MSG_SEARCH]).unwrap(), baseline);
    }
}
