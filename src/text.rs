use anyhow::Result;
use burn::tensor::{Device, Int, Tensor, TensorData, activation::silu, module};

use crate::{
    attention::{self, Kv, Segment},
    config::TextConfig,
    layers::{Linear, rms_norm},
    weights::Weights,
};

pub struct Decoder {
    tokens: Tensor<2>,
    layers: Vec<DecoderLayer>,
    norm: Tensor<1>,
    eps: f64,
    width: usize,
    head_dim: usize,
    theta: f32,
}

pub fn ids(values: Vec<i32>, device: &Device) -> Tensor<1, Int> {
    let length = values.len();
    Tensor::from_data(TensorData::new(values, [length]), device)
}

impl Decoder {
    pub fn load(w: &Weights<'_>, config: &TextConfig) -> Result<Self> {
        let width = config.hidden_size;
        Ok(Self {
            tokens: w.tensor(
                "language_model.model.embed_tokens.weight",
                [config.vocab_size, width],
            )?,
            layers: (0..config.num_hidden_layers)
                .map(|i| DecoderLayer::load(w, config, i))
                .collect::<Result<_>>()?,
            norm: w.tensor("language_model.model.norm.weight", [width])?,
            eps: config.rms_norm_eps,
            width,
            head_dim: config.head_dim,
            theta: config.rope_theta,
        })
    }

    pub fn embeddings(&self, input: Tensor<2, Int>) -> Tensor<3> {
        module::embedding(self.tokens.clone(), input)
    }

    pub fn logits(&self, hidden: Tensor<3>, candidate_ids: Vec<i32>) -> Tensor<3> {
        let indices = ids(candidate_ids, &hidden.device());
        module::linear(
            hidden,
            self.tokens.clone().select(0, indices).transpose(),
            None,
        )
    }

    pub fn prefix(&self, mut x: Tensor<3>, positions: &[usize], segments: &[Segment]) -> Vec<Kv> {
        let rope = self.rope(&x, positions);
        let mut cache = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, kv) = layer.forward(x, &rope, segments, None);
            cache.push(kv);
            x = next;
        }
        cache
    }

    pub fn suffix(
        &self,
        mut x: Tensor<3>,
        positions: &[usize],
        segments: &[Segment],
        cache: Vec<Kv>,
    ) -> Tensor<3> {
        let rope = self.rope(&x, positions);
        for (layer, kv) in self.layers.iter().zip(cache) {
            (x, _) = layer.forward(x, &rope, segments, Some(&kv));
        }
        rms_norm(x, &self.norm, self.eps)
    }

    fn rope(&self, x: &Tensor<3>, positions: &[usize]) -> Rope {
        let [b, t, width] = x.dims();
        debug_assert_eq!(width, self.width);
        debug_assert_eq!(b * t, positions.len());
        Rope::new(
            positions,
            [b, 1, t, self.head_dim / 2],
            self.theta,
            &x.device(),
            x.dtype(),
        )
    }
}

struct Rope {
    cos: Tensor<4>,
    sin: Tensor<4>,
    half: usize,
}

impl Rope {
    fn new(
        positions: &[usize],
        shape: [usize; 4],
        theta: f32,
        device: &Device,
        dtype: burn::tensor::DType,
    ) -> Self {
        let half = shape[3];
        let frequencies: Vec<_> = (0..half)
            .map(|index| theta.powf(-(index as f32) / half as f32))
            .collect();
        let mut sin = Vec::with_capacity(positions.len() * half);
        let mut cos = Vec::with_capacity(sin.capacity());
        for &position in positions {
            for &frequency in &frequencies {
                let (s, c) = (position as f32 * frequency).sin_cos();
                sin.push(s);
                cos.push(c);
            }
        }
        Self {
            half,
            cos: Tensor::<4>::from_data(TensorData::new(cos, shape), device).cast(dtype),
            sin: Tensor::<4>::from_data(TensorData::new(sin, shape), device).cast(dtype),
        }
    }

    fn apply(&self, x: Tensor<4>) -> Tensor<4> {
        let lo = x.clone().narrow(3, 0, self.half);
        let hi = x.narrow(3, self.half, self.half);
        let first = lo.clone() * self.cos.clone() - hi.clone() * self.sin.clone();
        let second = hi * self.cos.clone() + lo * self.sin.clone();
        Tensor::cat(vec![first, second], 3)
    }
}

struct DecoderLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    q_norm: Tensor<1>,
    k_norm: Tensor<1>,
    input_norm: Tensor<1>,
    post_norm: Tensor<1>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
}

impl DecoderLayer {
    fn load(w: &Weights<'_>, c: &TextConfig, index: usize) -> Result<Self> {
        let name = format!("language_model.model.layers.{index}");
        let width = c.hidden_size;
        let head = c.head_dim;
        let q_width = c.num_attention_heads * head;
        let kv_width = c.num_key_value_heads * head;
        let linear = |part: &str, input, output| {
            Linear::load(w, &format!("{name}.{part}"), input, output, false)
        };
        let norm = |part: &str, size| w.tensor(&format!("{name}.{part}.weight"), [size]);
        Ok(Self {
            q: linear("self_attn.q_proj", width, q_width)?,
            k: linear("self_attn.k_proj", width, kv_width)?,
            v: linear("self_attn.v_proj", width, kv_width)?,
            o: linear("self_attn.o_proj", q_width, width)?,
            gate: linear("mlp.gate_proj", width, c.intermediate_size)?,
            up: linear("mlp.up_proj", width, c.intermediate_size)?,
            down: linear("mlp.down_proj", c.intermediate_size, width)?,
            q_norm: norm("self_attn.q_norm", head)?,
            k_norm: norm("self_attn.k_norm", head)?,
            input_norm: norm("input_layernorm", width)?,
            post_norm: norm("post_attention_layernorm", width)?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: head,
            eps: c.rms_norm_eps,
        })
    }

    fn forward(
        &self,
        x: Tensor<3>,
        rope: &Rope,
        segments: &[Segment],
        prefix: Option<&Kv>,
    ) -> (Tensor<3>, Kv) {
        let [batch, frames, _] = x.dims();
        let h = rms_norm(x.clone(), &self.input_norm, self.eps);
        let project = |linear: &Linear, heads| {
            linear
                .forward(h.clone())
                .reshape([batch, frames, heads, self.head_dim])
                .swap_dims(1, 2)
        };
        let q = rope.apply(rms_norm(
            project(&self.q, self.heads),
            &self.q_norm,
            self.eps,
        ));
        let k = rope.apply(rms_norm(
            project(&self.k, self.kv_heads),
            &self.k_norm,
            self.eps,
        ));
        let v = project(&self.v, self.kv_heads);
        let kv = Kv { key: k, value: v };
        let context = attention::packed(q, &kv, prefix, segments, true)
            .swap_dims(1, 2)
            .reshape([batch, frames, self.heads * self.head_dim]);
        let x = x + self.o.forward(context);
        let h = rms_norm(x.clone(), &self.post_norm, self.eps);
        let h = silu(self.gate.forward(h.clone())) * self.up.forward(h);
        (x + self.down.forward(h), kv)
    }
}
