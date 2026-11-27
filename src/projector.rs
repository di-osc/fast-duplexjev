use anyhow::Result;
use burn::tensor::{Tensor, activation::silu};

use crate::{
    config::Config,
    layers::{Linear, rms_norm},
    weights::Weights,
};

pub(crate) struct Projector {
    pre: Tensor<1>,
    mid: Tensor<1>,
    first: Linear,
    second: Linear,
    inner: usize,
}

impl Projector {
    pub fn load(w: &Weights<'_>, c: &Config) -> Result<Self> {
        let input = c.audio_config.d_model * 2;
        let inner = c.hidden_size / 2;
        Ok(Self {
            pre: w.tensor("multi_modal_projector.ln_pre.weight", [input])?,
            mid: w.tensor("multi_modal_projector.ln_mid.weight", [inner])?,
            first: Linear::load(
                w,
                "multi_modal_projector.linear_1",
                input,
                c.hidden_size,
                false,
            )?,
            second: Linear::load(
                w,
                "multi_modal_projector.linear_2",
                inner,
                c.text_config.hidden_size,
                false,
            )?,
            inner,
        })
    }

    pub fn forward(&self, mut x: Tensor<3>) -> Tensor<3> {
        let [batch, frames, width] = x.dims();
        if frames % 2 != 0 {
            let padding = Tensor::zeros([batch, 1, width], (&x.device(), x.dtype()));
            x = Tensor::cat(vec![x, padding], 1);
        }
        let x = x.reshape([batch, frames.div_ceil(2), width * 2]);
        let x = self.first.forward(rms_norm(x, &self.pre, 1e-6));
        let value = x.clone().narrow(2, 0, self.inner);
        let gate = x.narrow(2, self.inner, self.inner);
        self.second
            .forward(rms_norm(value * silu(gate), &self.mid, 1e-6))
    }
}
