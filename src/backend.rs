use anyhow::{Result, bail};
use burn::tensor::{Device, FloatDType};
use clap::ValueEnum;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Backend {
    Flex,
    Metal,
    Cuda,
}

impl Backend {
    pub fn device(self, index: usize) -> Result<Device> {
        match self {
            #[cfg(feature = "flex")]
            Self::Flex if index == 0 => Ok(Device::flex()),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Self::Metal if index == 0 => Ok(Device::metal(burn::tensor::DeviceKind::DefaultDevice)),
            #[cfg(feature = "cuda")]
            Self::Cuda => Ok(Device::cuda(index)),
            _ => bail!("backend {self:?} or device index {index} is unavailable in this build"),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Precision {
    Auto,
    F32,
    Bf16,
    F16,
}

impl Precision {
    pub fn resolve(self, backend: Backend, device: &Device) -> Result<FloatDType> {
        let dtype = match self {
            Self::Auto if device.supports_dtype(FloatDType::BF16) => FloatDType::BF16,
            Self::Auto => FloatDType::F32,
            Self::F32 => FloatDType::F32,
            Self::Bf16 => FloatDType::BF16,
            Self::F16 => FloatDType::F16,
        };
        anyhow::ensure!(
            device.supports_dtype(dtype),
            "{backend:?} does not support {dtype:?}"
        );
        Ok(dtype)
    }
}
