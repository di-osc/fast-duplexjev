use std::{
    collections::{BTreeMap, HashSet},
    fmt,
};

use anyhow::{Result, ensure};
use blake2::{
    Blake2bVar,
    digest::{Update, VariableOutput},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    #[default]
    En,
    Zh,
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum ClipId {
    Name(String),
    Integer(i64),
}

impl fmt::Display for ClipId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(name) => formatter.write_str(name),
            Self::Integer(number) => number.fmt(formatter),
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum Target {
    One(ClipId),
    Many(Vec<ClipId>),
}

#[derive(Clone, Deserialize)]
pub struct Question {
    pub id: String,
    pub text: String,
    pub options: Vec<String>,
    #[serde(default)]
    pub lang: Language,
    pub audio: Option<Target>,
}

#[derive(Serialize)]
pub struct Answer {
    pub answer: String,
    pub confidence: f32,
    pub probs: BTreeMap<String, f32>,
}

pub type Answers = BTreeMap<String, Answer>;

impl Question {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.id.is_empty() && !self.text.is_empty(),
            "question id and text must be non-empty"
        );
        ensure!(
            (2..=26).contains(&self.options.len()),
            "question {:?} needs 2 to 26 options",
            self.id
        );
        ensure!(
            self.options.iter().collect::<HashSet<_>>().len() == self.options.len(),
            "question {:?} must have distinct options",
            self.id
        );
        Ok(())
    }

    pub(crate) fn render(&self, order: &[usize]) -> String {
        let (label, options, instruction) = match self.lang {
            Language::En => (
                "Question: ",
                "Options:",
                "Answer with only the letter of the correct option.",
            ),
            Language::Zh => ("问题：", "选项：", "请只回答正确选项的字母。"),
        };
        let mut text = format!("{label}{}\n\n{options}\n", self.text);
        for (letter, &option) in ('A'..='Z').zip(order) {
            text.push_str(&format!("{letter}. {}\n", self.options[option]));
        }
        text.push('\n');
        text.push_str(instruction);
        text
    }

    pub(crate) fn permutation(&self) -> Vec<usize> {
        let key = format!("{}\x1f{}\x1f0", self.id, self.options.join("\x1e"));
        let mut bytes = [0; 8];
        let mut hash = Blake2bVar::new(bytes.len()).expect("valid Blake2b output size");
        hash.update(key.as_bytes());
        hash.finalize_variable(&mut bytes)
            .expect("buffer matches output size");
        let mut rng = PythonShuffle::new(u64::from_be_bytes(bytes));
        let mut order: Vec<_> = (0..self.options.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        order
    }
}

// CPython's integer-seeded MT19937 and getrandbits() rejection sampling, used by
// the upstream Question.permutation(seed=0).
struct PythonShuffle {
    state: [u32; 624],
    index: usize,
}

impl PythonShuffle {
    fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = 19650218;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let words = [seed as u32, (seed >> 32) as u32];
        let count = if words[1] == 0 { 1 } else { 2 };
        let mut i = 1;
        for n in 0..624 {
            let j = n % count;
            let prev = state[i - 1];
            state[i] = (state[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525))
                .wrapping_add(words[j])
                .wrapping_add(j as u32);
            i += 1;
            if i == 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        for _ in 0..623 {
            let prev = state[i - 1];
            state[i] =
                (state[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i == 624 {
                state[0] = state[623];
                i = 1;
            }
        }
        state[0] = 0x80000000;
        Self { state, index: 624 }
    }

    fn below(&mut self, limit: usize) -> usize {
        let bits = limit.ilog2() + 1;
        loop {
            let sample = (self.next() >> (32 - bits)) as usize;
            if sample < limit {
                return sample;
            }
        }
    }

    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let x = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624] ^ (x >> 1) ^ ((x & 1) * 0x9908b0df);
            }
            self.index = 0;
        }
        let mut x = self.state[self.index];
        self.index += 1;
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680;
        x ^= (x << 15) & 0xefc60000;
        x ^ (x >> 18)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_order_matches_upstream_python() {
        let question = Question {
            id: "turn".into(),
            text: "Has the user finished?".into(),
            options: ["finished", "unfinished", "backchannel", "other"]
                .map(String::from)
                .to_vec(),
            lang: Language::En,
            audio: None,
        };
        assert_eq!(question.permutation(), [0, 2, 1, 3]);
    }
}
