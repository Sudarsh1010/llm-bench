# llm-bench

Benchmark token generation speed (tok/s) between [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) and [mistralrs](https://github.com/EricLBuehler/mistral.rs) using the same GGUF model and prompt.

## Quick Start (NixOS)

```bash
nix develop
cargo build --release
./target/release/llm-bench path/to/model.gguf
```

## Setup without Nix

Requires: Rust 1.88+, C/C++ compiler, cmake, libclang.

```bash
# Install system deps (Ubuntu/Debian)
sudo apt install libclang-dev build-essential cmake

# Build
cargo build --release

# Run
./target/release/llm-bench path/to/model.gguf
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `llama-cpp` | yes | Enable llama-cpp-2 backend |
| `mistralrs` | yes | Enable mistralrs backend |
| `cuda` | no | GPU acceleration (NVIDIA) |
| `metal` | no | GPU acceleration (Apple Silicon) |

```bash
# Build with CUDA
cargo build --features cuda --release

# Build only mistralrs (no libclang needed)
cargo build --no-default-features --features mistralrs --release

# Build only llama-cpp
cargo build --no-default-features --features llama-cpp --release
```

## Usage

```
llm-bench [OPTIONS] <MODEL>

ARGS:
    <MODEL>    Path to GGUF model file

OPTIONS:
    --backend <BACKEND>    llama-cpp, mistralrs, both [default: both]
    --gpu-layers <N>       GPU layers to offload (-1 = all) [default: -1]
    --ctx <N>              Context window size [default: 4096]
    --max-tokens <N>       Max tokens to generate [default: 256]
    --prompt <TEXT>        Prompt text
    --tok-model-id <ID>    HuggingFace tokenizer ID for mistralrs
    --seed <N>             Random seed [default: 1234]
```

### Examples

```bash
# Default: both backends, default prompt, 256 tokens
llm-bench ./model.gguf

# Only mistralrs, custom prompt, more tokens
llm-bench ./model.gguf --backend mistralrs --max-tokens 512 \
  --prompt "Write a haiku about Rust."

# Full GPU offload with CUDA
llm-bench ./model.gguf --gpu-layers 99 --features cuda

# Custom tokenizer model ID (for mistralrs)
llm-bench ./model.gguf --tok-model-id Qwen/Qwen2.5-3B-Instruct
```

## Output

```
=== LLM Bench Results ===
Model:       qwen2.5-3b-instruct-q4_k_m.gguf
Prompt:      "Explain quantum entanglement in 3 sentences."
Max Tokens:  256
GPU Layers:  -1
Context:     4096
Seed:        1234

┌─────────────┬───────────┬─────────────┬──────────────┬────────────┐
│ Backend     │ Tok/sec   │ Prompt(ms)  │ Generate(ms) │ Total(ms)  │
├─────────────┼───────────┼─────────────┼──────────────┼────────────┤
│ llama-cpp   │   42.50   │     120.3   │      6023.5  │    6143.8  │
  12 prompt tokens → 256 completion tokens (prompt: 99.7 tok/s)
│ mistralrs   │   38.75   │      95.2   │      6606.5  │    6701.7  │
  12 prompt tokens → 256 completion tokens (prompt: 126.0 tok/s)
└─────────────┴───────────┴─────────────┴──────────────┴────────────┘
```

```
=== LLM Bench Results ===
Model:       qwen2.5-3b-instruct-q4_k_m.gguf
Prompt:      "Explain quantum entanglement in 3 sentences."
Max Tokens:  256
GPU Layers:  -1
Context:     4096
Seed:        1234

┌─────────────┬───────────┬─────────────┬──────────────┬────────────┐
│ Backend     │ Tok/sec   │ Prompt(ms)  │ Generate(ms) │ Total(ms)  │
├─────────────┼───────────┼─────────────┼──────────────┼────────────┤
│ llama-cpp   │     64.45 │        50.2 │       1287.8 │     1338.0 │
  11 prompt tokens → 83 completion tokens (prompt: 219.0 tok/s)
│ mistralrs   │     59.67 │        68.0 │       1324.0 │     1392.0 │
  40 prompt tokens → 79 completion tokens (prompt: 588.2 tok/s)
└─────────────┴───────────┴─────────────┴──────────────┴────────────┘
```
