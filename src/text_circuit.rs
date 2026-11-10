//! Directly written weights for a small recurrent bulb-command circuit.
//! The vocabulary is a list of token identities; meaning is encoded by embeddings.

const STATE: usize = 5;
const FEATURES: usize = 6;
const INPUT: usize = STATE + FEATURES;
const FIRST: usize = 11;
const SECOND: usize = 10;

const VOCAB: &[&str] = &[
    "left",
    "first",
    "leftmost",
    "middle",
    "center",
    "centre",
    "second",
    "central",
    "right",
    "last",
    "rightmost",
    "third",
    "all",
    "everything",
    "every",
    "except",
    "but",
    "without",
    "not",
    "never",
    "dont",
    "off",
    "turn",
    "switch",
    "light",
    "illuminate",
    "activate",
    "on",
    "enable",
    "glow",
    "lit",
];

/// A hand-programmed recurrent ReLU network. No model weights are trained.
pub struct TextCircuit {
    embeddings: Vec<[f32; FEATURES]>,
    first: [[f32; INPUT]; FIRST],
    first_bias: [f32; FIRST],
    second: [[f32; FIRST]; SECOND],
    second_bias: [f32; SECOND],
    readout: [[f32; SECOND]; STATE],
}

impl Default for TextCircuit {
    fn default() -> Self {
        Self::new()
    }
}

impl TextCircuit {
    /// Write the lexicon and Boolean state transition directly into tensors.
    pub fn new() -> Self {
        let mut model = Self {
            embeddings: vec![[0.0; FEATURES]; VOCAB.len()],
            first: [[0.0; INPUT]; FIRST],
            first_bias: [0.0; FIRST],
            second: [[0.0; FIRST]; SECOND],
            second_bias: [0.0; SECOND],
            readout: [[0.0; SECOND]; STATE],
        };
        // Feature channels: left, middle, right, all, exclude, activation.
        for (word, feature) in [
            ("left", 0),
            ("first", 0),
            ("leftmost", 0),
            ("middle", 1),
            ("center", 1),
            ("centre", 1),
            ("second", 1),
            ("central", 1),
            ("right", 2),
            ("last", 2),
            ("rightmost", 2),
            ("third", 2),
            ("all", 3),
            ("everything", 3),
            ("every", 3),
            ("except", 4),
            ("but", 4),
            ("without", 4),
            ("not", 4),
            ("never", 4),
            ("dont", 4),
            ("off", 4),
            ("turn", 5),
            ("switch", 5),
            ("light", 5),
            ("illuminate", 5),
            ("activate", 5),
            ("on", 5),
            ("enable", 5),
            ("glow", 5),
            ("lit", 5),
        ] {
            let id = VOCAB
                .iter()
                .position(|candidate| *candidate == word)
                .unwrap();
            model.embeddings[id][feature] = 1.0;
        }
        // First ReLU layer computes the old flags, inclusions and exclusions.
        for i in 0..3 {
            model.first[i][i] = 1.0;
            model.first[3 + i][STATE + i] = 1.0;
            model.first[3 + i][3] = -1.0; // include = relu(position - exclusion mode)
            model.first[6 + i][STATE + i] = 1.0;
            model.first[6 + i][3] = 1.0;
            model.first_bias[6 + i] = -1.0; // remove = relu(position + exclusion mode - 1)
        }
        model.first[9][3] = 1.0;
        model.first[9][STATE + 4] = 1.0;
        model.first[10][4] = 1.0;
        model.first[10][STATE + 5] = 1.0;
        // Second ReLU layer computes clamp(old + include + all - remove, 0, 1).
        // Paired neurons implement clamp(x) = relu(x) - relu(x - 1).
        for i in 0..3 {
            for row in [2 * i, 2 * i + 1] {
                model.second[row][i] = 1.0;
                model.second[row][3 + i] = 1.0;
                model.second[row][6 + i] = -1.0;
            }
            model.second_bias[2 * i + 1] = -1.0;
            model.readout[i][2 * i] = 1.0;
            model.readout[i][2 * i + 1] = -1.0;
        }
        for (state, first_index) in [(3, 9), (4, 10)] {
            model.second[2 * state][first_index] = 1.0;
            model.second[2 * state + 1][first_index] = 1.0;
            model.second_bias[2 * state + 1] = -1.0;
            model.readout[state][2 * state] = 1.0;
            model.readout[state][2 * state + 1] = -1.0;
        }
        // 'all' needs to reach each state gate. It is carried through input feature 3
        // as part of the inclusion channels for each bulb, below in the first layer.
        for i in 0..3 {
            model.first[3 + i][STATE + 3] = 1.0;
        }
        model
    }

    /// Split text mechanically into lowercase alphanumeric word tokens; preserve order.
    pub fn tokens(text: &str) -> Vec<String> {
        text.to_lowercase()
            .replace("don't", "dont")
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_owned)
            .collect()
    }

    pub fn forward(&self, tokens: &[String]) -> [bool; 3] {
        let mut state = [0.0_f32; STATE];
        for token in tokens {
            let mut input = [0.0_f32; INPUT];
            input[..STATE].copy_from_slice(&state);
            if let Some(id) = VOCAB.iter().position(|word| *word == token) {
                input[STATE..].copy_from_slice(&self.embeddings[id]);
            }
            let mut first = [0.0_f32; FIRST];
            for i in 0..FIRST {
                first[i] = (self.first_bias[i]
                    + self.first[i]
                        .iter()
                        .zip(input)
                        .map(|(w, x)| w * x)
                        .sum::<f32>())
                .max(0.0);
            }
            let mut second = [0.0_f32; SECOND];
            for i in 0..SECOND {
                second[i] = (self.second_bias[i]
                    + self.second[i]
                        .iter()
                        .zip(first)
                        .map(|(w, x)| w * x)
                        .sum::<f32>())
                .max(0.0);
            }
            for i in 0..STATE {
                state[i] = self.readout[i].iter().zip(second).map(|(w, x)| w * x).sum();
            }
        }
        std::array::from_fn(|i| state[i] > 0.5 && state[4] > 0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::TextCircuit;
    #[test]
    fn programmed_text_examples() {
        let model = TextCircuit::new();
        for (text, expected) in [
            ("turn on the left and right bulbs", [true, false, true]),
            ("illuminate the first and last lights", [true, false, true]),
            (
                "switch on every bulb except the centre",
                [true, false, true],
            ),
            ("light all bulbs but the right", [true, true, false]),
            ("turn on the left and then the left", [true, false, false]),
            ("hello there", [false, false, false]),
        ] {
            assert_eq!(
                model.forward(&TextCircuit::tokens(text)),
                expected,
                "{text}"
            );
        }
    }
}
