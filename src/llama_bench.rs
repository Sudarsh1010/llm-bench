#[cfg(feature = "llama-cpp")]
use std::num::NonZeroU32;
#[cfg(feature = "llama-cpp")]
use std::pin::pin;
#[cfg(feature = "llama-cpp")]
use std::time::Instant;

#[cfg(feature = "llama-cpp")]
use anyhow::Result;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::context::params::LlamaContextParams;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::llama_backend::LlamaBackend;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::llama_batch::LlamaBatch;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::model::params::LlamaModelParams;
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::model::{AddBos, LlamaModel};
#[cfg(feature = "llama-cpp")]
use llama_cpp_2::sampling::LlamaSampler;

#[cfg(feature = "llama-cpp")]
use crate::{BenchConfig, BenchResult};

pub struct LlamaBenchOutput {
    pub result: BenchResult,
    pub generated_text: String,
}

#[cfg(feature = "llama-cpp")]
pub fn bench_llama(config: &BenchConfig) -> Result<LlamaBenchOutput> {
    let backend = LlamaBackend::init()?;

    let model_params =
        pin!(LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers as u32));
    println!("  Loading GGUF model from {}...", config.model_path);
    let model = LlamaModel::load_from_file(&backend, &config.model_path, &model_params)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(config.n_ctx));
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .map_err(|e| anyhow::anyhow!("Failed to create context: {e}"))?;

    println!("  Tokenizing prompt...");
    let tokens = model
        .str_to_token(&config.prompt, AddBos::Always)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize: {e}"))?;
    let n_prompt = tokens.len();

    println!("  Processing {n_prompt} prompt tokens...");
    let mut batch = LlamaBatch::new(512, 1);
    for (i, &token) in tokens.iter().enumerate() {
        let is_last = i == n_prompt - 1;
        batch.add(token, i as i32, &[0], is_last)?;
    }

    let prompt_start = Instant::now();
    ctx.decode(&mut batch)
        .map_err(|e| anyhow::anyhow!("Prompt decode failed: {e}"))?;
    let prompt_elapsed = prompt_start.elapsed().as_secs_f64() * 1000.0;

    println!("  Generating up to {} tokens...", config.max_tokens);

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(config.seed as u32),
        LlamaSampler::greedy(),
    ]);

    let mut generated_text = String::new();
    let mut n_cur = n_prompt as i32;
    let mut n_generated: usize = 0;

    let gen_start = Instant::now();

    for _ in 0..config.max_tokens {
        let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(new_token);

        if model.is_eog_token(new_token) {
            break;
        }

        let token_str = model
            .token_to_piece(new_token, &mut encoding_rs::UTF_8.new_decoder(), true, None)
            .unwrap_or_default();
        generated_text.push_str(&token_str);

        batch.clear();
        batch.add(new_token, n_cur, &[0], true)?;
        n_cur += 1;
        n_generated += 1;

        ctx.decode(&mut batch)
            .map_err(|e| anyhow::anyhow!("Generation decode failed: {e}"))?;
    }

    let gen_elapsed = gen_start.elapsed().as_secs_f64() * 1000.0;
    let total_elapsed = prompt_elapsed + gen_elapsed;

    let tok_per_sec = if gen_elapsed > 0.0 {
        n_generated as f64 / (gen_elapsed / 1000.0)
    } else {
        0.0
    };
    let prompt_tok_per_sec = if prompt_elapsed > 0.0 {
        n_prompt as f64 / (prompt_elapsed / 1000.0)
    } else {
        0.0
    };

    Ok(LlamaBenchOutput {
        result: BenchResult {
            backend: "llama-cpp".to_string(),
            prompt_tokens: n_prompt,
            completion_tokens: n_generated,
            prompt_time_ms: prompt_elapsed,
            generate_time_ms: gen_elapsed,
            total_time_ms: total_elapsed,
            tok_per_sec,
            prompt_tok_per_sec,
        },
        generated_text,
    })
}
