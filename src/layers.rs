use anyhow::Result;
use burn::tensor::{FloatDType, Tensor, module};

use crate::weights::Weights;

pub struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}

impl Linear {
    pub fn load(
        weights: &Weights<'_>,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: weights
                .tensor::<2>(&format!("{name}.weight"), [output, input])?
                .transpose(),
            bias: bias
                .then(|| weights.tensor(&format!("{name}.bias"), [output]))
                .transpose()?,
        })
    }

    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        module::linear(input, self.weight.clone(), self.bias.clone())
    }
}

pub struct LayerNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
}

impl LayerNorm {
    pub fn load(weights: &Weights<'_>, name: &str, width: usize) -> Result<Self> {
        Ok(Self {
            weight: weights.tensor(&format!("{name}.weight"), [width])?,
            bias: weights.tensor(&format!("{name}.bias"), [width])?,
        })
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        module::layer_norm(x, self.weight.clone(), Some(self.bias.clone()), 1e-5)
    }
}

pub fn rms_norm<const D: usize>(x: Tensor<D>, weight: &Tensor<1>, eps: f64) -> Tensor<D> {
    let dtype = x.dtype();
    let x = x.cast(FloatDType::F32);
    let variance = (x.clone() * x.clone()).mean_dim(D - 1);
    let width = weight.dims()[0];
    let mut shape = [1; D];
    shape[D - 1] = width;
    (x / (variance + eps).sqrt()).cast(dtype) * weight.clone().reshape(shape)
}
