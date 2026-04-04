#[cfg(feature = "mistralrs")]
use std::time::Instant;

#[cfg(feature = "mistralrs")]
use cuda_runtime_sys::{cudaFree, cudaGetDevice, cudaMalloc, cudaSetDevice, cudaStreamDestroy, cudaStream_t, cudaStreamCreate, cudaDeviceSynchronize, cudaGetLastError, cudaDeviceProp, cudaGetDeviceProperties, cudaError_t, cudaSuccess};
#[cfg(feature = "mistralrs")]
use std::ptr::null_mut;

#[cfg(feature = "mistralrs")]
use anyhow::Result;
#[cfg(feature = "mistralrs")]
use mistralrs::{
    GgufModelBuilder, MemoryGpuConfig, PagedAttentionMetaBuilder, RequestBuilder,
    TextMessageRole,
};

#[cfg(feature = "mistralrs")]
use crate::{BenchConfig, BenchResult};

pub struct MistralBenchOutput {
    pub result: BenchResult,
    pub generated_text: String,
}

#[cfg(feature = "mistralrs")]
pub async fn bench_mistral(config: &BenchConfig) -> Result<MistralBenchOutput> {
    // Initialize CUDA context
    unsafe {
        // Set the CUDA device (using device 0 as default)
        let mut device = 0;
        let result = cudaSetDevice(device);
        if result != cudaSuccess {
            return Err(anyhow::anyhow!("Failed to set CUDA device: {}", cudaGetLastError()));
        }

        // Get device properties to verify CUDA capability
        let mut props: cudaDeviceProp = std::mem::zeroed();
        let result = cudaGetDeviceProperties(&mut props, device);
        if result != cudaSuccess {
            return Err(anyhow::anyhow!("Failed to get device properties: {}", cudaGetLastError()));
        }

        println!("  Using CUDA device: {} ({}), Compute Capability: {}.{}", 
                device, 
                std::ffi::CStr::from_ptr(props.name.as_ptr()).to_string_lossy(),
                props.major, props.minor);
    }

    let model_path = std::path::Path::new(&config.model_path);
    let model_dir = model_path
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid model directory path"))?;
    let model_file = model_path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("No filename in model path"))?
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid model filename"))?;

    println!("  Building GGUF model from directory: {model_dir}, file: {model_file}");

    let mut builder = GgufModelBuilder::new(model_dir, vec![model_file.to_string()]);

    if let Some(tok_id) = &config.tok_model_id {
        println!("  Using tokenizer model ID: {tok_id}");
        builder = builder.with_tok_model_id(tok_id.as_str());
    }

    builder = builder.with_logging();

    builder = builder.with_paged_attn(|| {
        PagedAttentionMetaBuilder::default()
            .with_gpu_memory(MemoryGpuConfig::ContextSize(config.n_ctx as usize))
            .build()
    })?;

    println!("  Loading model (this may take a moment)...");
    let wall_start = Instant::now();
    let model = builder.build().await?;
    let load_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Model loaded in {load_ms:.0} ms");

    let request = RequestBuilder::new()
        .add_message(TextMessageRole::User, &config.prompt)
        .set_sampler_max_len(config.max_tokens as usize);

    println!("  Sending chat request...");
    let request_start = Instant::now();
    let response = model.send_chat_request(request).await?;
    let total_wall_ms = request_start.elapsed().as_secs_f64() * 1000.0;

    let content = response
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    let usage = response.usage;

    let prompt_time_ms = usage.total_prompt_time_sec as f64 * 1000.0;
    let gen_time_ms = usage.total_completion_time_sec as f64 * 1000.0;
    let total_time_ms = usage.total_time_sec as f64 * 1000.0;

    let tok_per_sec = usage.avg_compl_tok_per_sec as f64;
    let prompt_tok_per_sec = usage.avg_prompt_tok_per_sec as f64;

    println!("  Wall-clock total: {total_wall_ms:.0} ms, reported total: {total_time_ms:.0} ms");

    Ok(MistralBenchOutput {
        result: BenchResult {
            backend: "mistralrs".to_string(),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            prompt_time_ms,
            generate_time_ms: gen_time_ms,
            total_time_ms,
            tok_per_sec,
            prompt_tok_per_sec,
        },
        generated_text: content,
    })
}
