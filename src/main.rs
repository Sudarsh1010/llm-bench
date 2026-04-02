use anyhow::{bail, Result};
use clap::Parser;

#[cfg(feature = "llama-cpp")]
mod llama_bench;

#[cfg(feature = "mistralrs")]
mod mistral_bench;

// ── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Clone, clap::ValueEnum)]
enum Backend {
    #[value(name = "llama-cpp")]
    LlamaCpp,
    #[value(name = "mistralrs")]
    Mistralrs,
    Both,
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Backend::LlamaCpp => write!(f, "llama-cpp"),
            Backend::Mistralrs => write!(f, "mistralrs"),
            Backend::Both => write!(f, "both"),
        }
    }
}

#[derive(Parser)]
#[command(name = "llm-bench", about = "Benchmark LLM inference backends")]
struct Cli {
    /// Path to GGUF model file
    model: String,

    /// Backend to benchmark: llama-cpp, mistralrs, both
    #[arg(long, default_value = "both")]
    backend: Backend,

    /// Number of GPU layers (-1 = all)
    #[arg(long, default_value_t = -1)]
    gpu_layers: i32,

    /// Context size
    #[arg(long, default_value_t = 4096)]
    ctx: u32,

    /// Max generation tokens
    #[arg(long, default_value_t = 256)]
    max_tokens: u32,

    /// Prompt text
    #[arg(long, default_value = "Explain quantum entanglement in 3 sentences.")]
    prompt: String,

    /// HuggingFace tokenizer model ID for mistralrs
    #[arg(long)]
    tok_model_id: Option<String>,

    /// Random seed
    #[arg(long, default_value_t = 1234)]
    seed: u64,
}

// ── Shared config ────────────────────────────────────────────────────────────

struct BenchConfig {
    model_path: String,
    n_gpu_layers: i32,
    n_ctx: u32,
    prompt: String,
    max_tokens: u32,
    seed: u64,
    tok_model_id: Option<String>,
}

impl From<Cli> for BenchConfig {
    fn from(cli: Cli) -> Self {
        Self {
            model_path: cli.model,
            n_gpu_layers: cli.gpu_layers,
            n_ctx: cli.ctx,
            prompt: cli.prompt,
            max_tokens: cli.max_tokens,
            seed: cli.seed,
            tok_model_id: cli.tok_model_id,
        }
    }
}

// ── Benchmark result ─────────────────────────────────────────────────────────

struct BenchResult {
    backend: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_time_ms: f64,
    generate_time_ms: f64,
    total_time_ms: f64,
    tok_per_sec: f64,
    prompt_tok_per_sec: f64,
}

// ── Table printing ───────────────────────────────────────────────────────────

fn print_results(config: &BenchConfig, results: &[BenchResult]) {
    let model_name = std::path::Path::new(&config.model_path)
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| config.model_path.clone());

    println!("\n=== LLM Bench Results ===");
    println!("Model:       {model_name}");
    println!(
        "Prompt:      \"{}\"",
        if config.prompt.len() > 60 {
            format!("{}...", &config.prompt[..60])
        } else {
            config.prompt.clone()
        }
    );
    println!("Max Tokens:  {}", config.max_tokens);
    println!("GPU Layers:  {}", config.n_gpu_layers);
    println!("Context:     {}", config.n_ctx);
    println!("Seed:        {}", config.seed);

    // Column widths: Backend(13) | Tok/sec(11) | Prompt(ms)(13) | Generate(ms)(14) | Total(ms)(12)
    println!();
    println!("┌─────────────┬───────────┬─────────────┬──────────────┬────────────┐");
    println!("│ Backend     │ Tok/sec   │ Prompt(ms)  │ Generate(ms) │ Total(ms)  │");
    println!("├─────────────┼───────────┼─────────────┼──────────────┼────────────┤");

    for r in results {
        println!(
            "│ {:<11} │ {:>9.2} │ {:>11.1} │ {:>12.1} │ {:>10.1} │",
            r.backend, r.tok_per_sec, r.prompt_time_ms, r.generate_time_ms, r.total_time_ms
        );
        println!(
            "  {} prompt tokens → {} completion tokens (prompt: {:.1} tok/s)",
            r.prompt_tokens, r.completion_tokens, r.prompt_tok_per_sec
        );
    }

    println!("└─────────────┴───────────┴─────────────┴──────────────┴────────────┘");
}

// ── Tokenizer model ID inference ─────────────────────────────────────────────

fn infer_tok_model_id(model_path: &str) -> Option<String> {
    let filename = std::path::Path::new(model_path)
        .file_name()?
        .to_str()?
        .to_lowercase();

    // Strip common quantization suffixes: -q4_k_m, -q5_0, -q8_0, -q4_0, etc.
    let base = regex_quant_suffix(&filename);

    // Qwen models: qwen2.5-3b-instruct-q4_k_m.gguf -> Qwen/Qwen2.5-3B-Instruct
    if let Some(rest) = base.strip_prefix("qwen2.5-") {
        if let Some(size) = extract_size(rest) {
            let instruct = if rest.contains("instruct") {
                "-Instruct"
            } else {
                ""
            };
            return Some(format!("Qwen/Qwen2.5-{size}{instruct}"));
        }
    }

    if let Some(rest) = base.strip_prefix("qwen2-") {
        if let Some(size) = extract_size(rest) {
            let instruct = if rest.contains("instruct") {
                "-Instruct"
            } else {
                ""
            };
            return Some(format!("Qwen/Qwen2-{size}{instruct}"));
        }
    }

    // Llama models: llama-3.1-8b-instruct -> meta-llama/Meta-Llama-3.1-8B-Instruct
    if let Some(rest) = base.strip_prefix("llama-") {
        if let Some(size) = extract_size(rest) {
            let instruct = if rest.contains("instruct") {
                "-Instruct"
            } else {
                ""
            };
            return Some(format!("meta-llama/Meta-Llama-{size}{instruct}"));
        }
    }

    // Mistral models: mistral-7b-instruct -> mistralai/Mistral-7B-Instruct-v0.3
    if let Some(rest) = base.strip_prefix("mistral-") {
        if let Some(size) = extract_size(rest) {
            let instruct = if rest.contains("instruct") {
                "-Instruct-v0.3"
            } else {
                "-v0.3"
            };
            return Some(format!("mistralai/Mistral-{size}{instruct}"));
        }
    }

    // Phi models: phi-3.5-mini-instruct -> microsoft/Phi-3.5-mini-instruct
    if let Some(rest) = base.strip_prefix("phi-") {
        return Some(format!("microsoft/Phi-{rest}"));
    }

    None
}

/// Strip quantization suffix like -q4_k_m, -q5_0, etc.
fn regex_quant_suffix(s: &str) -> String {
    let s = s.strip_suffix(".gguf").unwrap_or(s);

    // Match patterns like: -q4_k_m, -q5_0, -q8_0, -q4_0, -q2_k, -q3_k_s, -q6_k
    let re_suffixes = [
        "-q8_0", "-q6_k", "-q5_k_m", "-q5_k_s", "-q5_0", "-q4_k_m", "-q4_k_s", "-q4_0",
        "-q3_k_m", "-q3_k_s", "-q3_k_l", "-q2_k", "-q4", "-q5", "-q8",
    ];

    for suffix in &re_suffixes {
        if let Some(base) = s.strip_suffix(suffix) {
            return base.to_string();
        }
    }

    s.to_string()
}

/// Extract size like "3b", "7b", "8b", "14b", "70b" from the beginning of a string
fn extract_size(s: &str) -> Option<String> {
    let s_upper = s.to_uppercase();
    for size in ["72B", "70B", "32B", "14B", "8B", "7B", "3B", "1.5B", "1B", "0.5B"] {
        if s_upper.starts_with(size) {
            return Some(size.to_string());
        }
    }
    None
}

// ── Run benchmarks ───────────────────────────────────────────────────────────

fn run_benchmarks(config: &BenchConfig, backend: &Backend) -> Result<Vec<BenchResult>> {
    let mut results = Vec::new();
    let _config = config;

    if matches!(backend, Backend::LlamaCpp | Backend::Both) {
        #[cfg(feature = "llama-cpp")]
        {
            println!("\n[1/{}] Loading model with llama-cpp-2...",
                if matches!(backend, Backend::Both) { 2 } else { 1 });
            let result = llama_bench::bench_llama(config)?;
            println!("  Generated text (first 80 chars): {:.80}", result.generated_text);
            results.push(result.result);
        }
        #[cfg(not(feature = "llama-cpp"))]
        {
            bail!("llama-cpp backend not compiled. Rebuild with: cargo build --features llama-cpp");
        }
    }

    if matches!(backend, Backend::Mistralrs | Backend::Both) {
        #[cfg(feature = "mistralrs")]
        {
            println!("\n[2/2] Loading model with mistralrs...");
            let rt = tokio::runtime::Runtime::new()?;
            let result = rt.block_on(mistral_bench::bench_mistral(config))?;
            println!("  Generated text (first 80 chars): {:.80}", result.generated_text);
            results.push(result.result);
        }
        #[cfg(not(feature = "mistralrs"))]
        {
            bail!("mistralrs backend not compiled. Rebuild with: cargo build --features mistralrs");
        }
    }

    Ok(results)
}

// ── Entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();

    if !std::path::Path::new(&cli.model).exists() {
        bail!("Model file not found: {}", cli.model);
    }

    let backend = cli.backend.clone();
    let mut config: BenchConfig = cli.into();

    if config.tok_model_id.is_none() {
        config.tok_model_id = infer_tok_model_id(&config.model_path);
        if let Some(ref id) = config.tok_model_id {
            println!("Auto-inferred tokenizer model ID: {id}");
        } else {
            eprintln!("Warning: Could not auto-infer tokenizer model ID from filename.");
            eprintln!("         Mistralrs may fail. Use --tok-model-id to specify manually.");
        }
    }

    let results = run_benchmarks(&config, &backend)?;
    print_results(&config, &results);

    Ok(())
}
