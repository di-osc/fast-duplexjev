use std::ops::Range;

use burn::tensor::{Tensor, module, ops::AttentionModuleOptions};

/// A contiguous run of queries and, for a question, the clip's shared prefix.
pub(crate) struct Segment {
    pub tokens: Range<usize>,
    pub prefix: Option<Range<usize>>,
}

pub(crate) struct Kv {
    pub key: Tensor<4>,
    pub value: Tensor<4>,
}

/// All projections operate on packed tokens. Each attention dispatch sees only
/// its own segment and optional prefix, so there is no batch padding or dense
/// cross-segment mask. The cache holds one copy of each clip's prefix.
pub(crate) fn packed(
    query: Tensor<4>,
    kv: &Kv,
    prefix: Option<&Kv>,
    segments: &[Segment],
    causal: bool,
) -> Tensor<4> {
    let slice = |x: &Tensor<4>, range: &Range<usize>| x.clone().narrow(2, range.start, range.len());
    let join = |current: &Tensor<4>, cached: Option<&Tensor<4>>, segment: &Segment| {
        let own = slice(current, &segment.tokens);
        match &segment.prefix {
            Some(range) => Tensor::cat(vec![slice(cached.expect("prefix KV"), range), own], 2),
            None => own,
        }
    };
    let options = AttentionModuleOptions {
        is_causal: causal,
        ..Default::default()
    };
    let output = segments
        .iter()
        .map(|segment| {
            let q = slice(&query, &segment.tokens);
            let k = join(&kv.key, prefix.map(|p| &p.key), segment);
            let v = join(&kv.value, prefix.map(|p| &p.value), segment);
            let [batch, heads, frames, width] = q.dims();
            let kv_heads = k.dims()[1];
            if heads == kv_heads {
                return module::attention(q, k, v, None, None, options);
            }

            // Burn 0.22's attention fallback broadcasts singleton heads.
            // Move KV heads into the batch dimension so each KV head can serve
            // its matching query group without duplicating keys and values.
            debug_assert_eq!(batch, 1);
            let q = q.reshape([kv_heads, heads / kv_heads, frames, width]);
            let value_width = v.dims()[3];
            module::attention(q, k.swap_dims(0, 1), v.swap_dims(0, 1), None, None, options)
                .reshape([batch, heads, frames, value_width])
        })
        .collect::<Vec<_>>();
    Tensor::cat(output, 2)
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, TensorData};

    use super::*;

    fn check(device: Device) {
        let tensor = |values: Vec<f32>, heads, frames| {
            Tensor::<4>::from_data(TensorData::new(values, [1, heads, frames, 1]), &device)
        };
        // Two question segments of different lengths use the same two-token prefix.
        // Zero queries make the expected output the mean of the visible values.
        let prefix = Kv {
            key: tensor(vec![0.; 4], 2, 2),
            value: tensor(vec![2., 4., 4., 8.], 2, 2),
        };
        let suffix = Kv {
            key: tensor(vec![0.; 6], 2, 3),
            value: tensor(vec![12., 24., 30., 24., 48., 60.], 2, 3),
        };
        let segments = [
            Segment {
                tokens: 0..1,
                prefix: Some(0..2),
            },
            Segment {
                tokens: 1..3,
                prefix: Some(0..2),
            },
        ];
        let actual = packed(
            tensor(vec![0.; 12], 4, 3),
            &suffix,
            Some(&prefix),
            &segments,
            true,
        )
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
        for (actual, expected) in actual
            .into_iter()
            .zip([6., 10., 15., 6., 10., 15., 12., 20., 30., 12., 20., 30.])
        {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[cfg(feature = "flex")]
    #[test]
    fn packed_attention_does_not_mix_questions_on_flex() {
        check(Device::flex());
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn packed_attention_does_not_mix_questions_on_metal() {
        check(Device::metal(burn::tensor::DeviceKind::DefaultDevice));
    }
}
