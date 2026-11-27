use std::{collections::HashMap, fs::File, path::Path};

use anyhow::{Context, Result, bail, ensure};
use burn::tensor::{Device, FloatDType, Tensor, TensorData};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

#[derive(Deserialize)]
struct Index {
    weight_map: HashMap<String, String>,
}

pub struct Weights<'a> {
    index: HashMap<String, usize>,
    shards: Vec<Mmap>,
    device: &'a Device,
    dtype: FloatDType,
}

impl<'a> Weights<'a> {
    pub fn open(path: &Path, device: &'a Device, dtype: FloatDType) -> Result<Self> {
        let index: Index =
            serde_json::from_reader(File::open(path.join("model.safetensors.index.json"))?)?;
        let mut files = HashMap::new();
        let mut shards = Vec::new();
        let mut names = HashMap::new();
        for (name, shard) in index.weight_map {
            ensure!(
                Path::new(&shard)
                    .file_name()
                    .is_some_and(|basename| basename == shard.as_str()),
                "invalid checkpoint shard name {shard:?}"
            );
            let id = match files.get(&shard) {
                Some(&id) => id,
                None => {
                    let file =
                        File::open(path.join(&shard)).with_context(|| format!("open {shard}"))?;
                    // The checkpoint must remain immutable while it is being loaded.
                    let map = unsafe { Mmap::map(&file) }?;
                    SafeTensors::read_metadata(&map).with_context(|| format!("read {shard}"))?;
                    let id = shards.len();
                    shards.push(map);
                    files.insert(shard, id);
                    id
                }
            };
            names.insert(name, id);
        }
        Ok(Self {
            index: names,
            shards,
            device,
            dtype,
        })
    }

    pub fn with_dtype(mut self, dtype: FloatDType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn tensor<const D: usize>(&self, name: &str, shape: [usize; D]) -> Result<Tensor<D>> {
        let &id = self
            .index
            .get(name)
            .with_context(|| format!("missing weight {name}"))?;
        let weights = SafeTensors::deserialize(&self.shards[id])?;
        let view = weights.tensor(name)?;
        ensure!(
            view.shape() == shape,
            "{name} has shape {:?}, expected {shape:?}",
            view.shape()
        );
        ensure!(
            view.dtype() == Dtype::BF16,
            "{name} must use BF16 checkpoint storage"
        );
        let data = match self.dtype {
            FloatDType::BF16 => {
                TensorData::from_bytes_vec(view.data().to_vec(), shape, self.dtype.into())
            }
            FloatDType::F32 => TensorData::new(
                bf16_values(view.data())
                    .map(bf16::to_f32)
                    .collect::<Vec<_>>(),
                shape,
            ),
            FloatDType::F16 => TensorData::new(
                bf16_values(view.data())
                    .map(|value| f16::from_f32(value.to_f32()))
                    .collect::<Vec<_>>(),
                shape,
            ),
            _ => bail!("unsupported weight dtype {:?}", self.dtype),
        };
        Ok(Tensor::from_data(data, (self.device, self.dtype.into())))
    }
}

fn bf16_values(data: &[u8]) -> impl Iterator<Item = bf16> + '_ {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|&bytes| bf16::from_bits(u16::from_le_bytes(bytes)))
}

#[cfg(all(test, feature = "flex"))]
mod tests {
    use memmap2::MmapMut;
    use safetensors::tensor::{TensorView, serialize};

    use super::*;

    #[test]
    fn weights_use_requested_tensor_dtype() {
        let bf16_bytes = bf16::from_f32(1.5).to_bits().to_le_bytes();
        let tensor = TensorView::new(Dtype::BF16, vec![1], &bf16_bytes).unwrap();
        let bytes = serialize([("weight", tensor)], None).unwrap();
        let mut shard = MmapMut::map_anon(bytes.len()).unwrap();
        shard.copy_from_slice(&bytes);
        let device = Device::flex();
        let mut weights = Weights {
            index: HashMap::from([("weight".into(), 0)]),
            shards: vec![shard.make_read_only().unwrap()],
            device: &device,
            dtype: FloatDType::BF16,
        };
        for dtype in [FloatDType::BF16, FloatDType::F16, FloatDType::F32] {
            weights = weights.with_dtype(dtype);
            let tensor = weights.tensor::<1>("weight", [1]).unwrap();
            assert_eq!(tensor.dtype(), dtype.into());
        }
    }
}
