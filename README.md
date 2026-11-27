# DuplexJev in Rust

DuplexJev-4B-Para v1.4 speech decisions on Burn. The command-line binary is
`duplexjev`. It loads the original sharded Hugging Face Safetensors checkpoint
and accepts WAV or FLAC audio.

## Install and run

From the project root, install the `duplexjev` binary with Metal and Flex on Apple Silicon:

```sh
cargo install --locked --path . --bin duplexjev --features metal
duplexjev serve
```

For a CPU-only build, omit `--features metal` and use `--device flex`. For CUDA,
build on an NVIDIA system with `--no-default-features --features cuda` and use
`--device cuda`. The CUDA feature compiles on macOS, but CUDA execution requires
an NVIDIA driver and device.

Omit `--model` to download the pinned v1.4 checkpoint from Hugging Face via
`modelhub`. It downloads only the configuration, tokenizer, weight index and
listed shards, then reuses them from the modelhub cache. Pass `--cache-dir DIR`
to override the cache directory. `--revision` overrides the revision for a model
ID. A local directory passed to `--model` is loaded directly and must contain
`config.json`, `tokenizer.json`, the weight index, and both Safetensors shards.
The weights occupy about 8.44 GB on disk. The default
Qwen3 precision is BF16 on devices that support it and FP32 otherwise. In BF16
mode, audio encoder and projector weights and computation use FP32, then a
single cast converts the projected audio embeddings to BF16 for Qwen3. Qwen3
weights and its KV cache use BF16. Use `--precision f32` to run the entire model
in FP32 or `--precision f16` to request FP16.

```sh
duplexjev decide call.wav --q 'turn|Has the user finished?|finished,unfinished,backchannel'
duplexjev batch --audio a=call.wav --audio b=other.wav --questions questions.json
duplexjev serve --cache-dir "$HOME/models/duplexjev"
duplexjev serve --help
```

Question files are JSON arrays of objects with `id`, `text`, `options`, and
optional `lang` (`en` or `zh`). For batch decisions, set `audio` to a clip ID,
an array of IDs, or `"*"` for every clip.

## HTTP API

The server binds to `127.0.0.1:8000` by default and groups requests every 160 ms.
`--max-items` caps clips per inference batch. `--max-tokens` controls how many
packed prefix and question tokens are grouped together. A single clip over
the token budget runs by itself.

- `GET /health` returns the tick interval and number of nonempty ticks.
- `POST /v1/decide` takes `audio_b64`, `questions`, and optional `context` or
  `lang`. The response's `answers` map is keyed by question ID.
- `POST /v1/decide_batch` takes `audios` (clip ID to Base64 audio), `questions`
  with clip selectors, and optional `context` (clip ID to text) or `lang`.
  The response's `answers` map is keyed first by clip ID and then by question ID.

Each answer contains the winning option, its confidence, and all option
probabilities. The server also includes upstream-compatible batching and
timing fields.

## Modules

- `audio` decodes audio, mixes to mono, resamples, and computes 128-bin log-Mel.
- `encoder` implements the Qwen3-ASR audio tower.
- `projector` stacks and projects encoder frames to the Qwen3 embedding width.
- `model` assembles audio and text embeddings and scores the answers.
  `question` formats prompts and reproduces upstream option ordering.
- `text` implements Qwen3 with RoPE, grouped-query attention, and the tied
  output head. `weights` loads the original checkpoint tensors.
- `attention` runs the variable-length sequence segments. Projection and MLP
  operations see packed tokens, while attention uses one Burn call per segment.
  Each query segment sees its own causal history and its clip's shared prefix.
  KV storage is allocated from the actual prefix lengths for each batch and is
  dropped after scoring. At 4B, prefix KV uses 288 KiB per cached token in FP32
  or 144 KiB at 16-bit precision, before temporary attention buffers.
- `backend`, `server`, and the binary handle device selection, HTTP batching,
  and commands.

Model weights are distributed under their own license.
[DuplexJev-4B-Para model card](https://huggingface.co/adventists-ai/DuplexJev-4B-Para)
