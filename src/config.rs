use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub stack_factor: usize,
    pub audio_full_length: bool,
    pub hidden_size: usize,
    pub projector_act: String,
    pub projector_ln_mid: bool,
    pub audio_config: AudioConfig,
    pub text_config: TextConfig,
}

#[derive(Deserialize)]
pub struct AudioConfig {
    pub model_type: String,
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub encoder_ffn_dim: usize,
    pub downsample_hidden_size: usize,
    pub num_mel_bins: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
}

#[derive(Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        let a = &self.audio_config;
        let t = &self.text_config;
        ensure!(
            self.architectures
                .iter()
                .any(|name| name == "DuplexJevForConditionalGeneration")
                && self.stack_factor == 2
                && self.audio_full_length
                && self.hidden_size > 0
                && self.hidden_size.is_multiple_of(2)
                && self.projector_act == "swiglu"
                && self.projector_ln_mid
                && a.model_type == "qwen3_asr_encoder_portable"
                && a.n_window == 50
                && a.n_window_infer == 800
                && a.num_mel_bins == 128
                && a.d_model.is_multiple_of(2)
                && a.encoder_attention_heads > 0
                && a.d_model.is_multiple_of(a.encoder_attention_heads)
                && t.model_type == "qwen3"
                && t.tie_word_embeddings
                && t.head_dim > 0
                && t.head_dim.is_multiple_of(2)
                && t.num_key_value_heads > 0
                && t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
            "unsupported checkpoint configuration (expected DuplexJev-4B-Para v1.4 architecture)"
        );
        Ok(())
    }
}
