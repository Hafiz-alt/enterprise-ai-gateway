// =============================================================================
// src/bin/pre_download.rs — Docker Build-Time Model Pre-Downloader
// =============================================================================
//
// Executed strictly during Docker image compilation to download and verify
// the FastEmbed ONNX weights and tokenizer files into a target cache folder.
//
// This guarantees the resulting Docker image is completely self-contained and
// immutable — the production container will never touch Hugging Face at runtime.
// =============================================================================

use std::path::PathBuf;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("============================================================");
    println!("🚀 Pre-downloading FastEmbed AllMiniLML6V2 Model for Docker");
    println!("============================================================");

    let cache_dir_str = std::env::var("FASTEMBED_CACHE_PATH")
        .or_else(|_| std::env::var("FASTEMBED_CACHE_DIR"))
        .unwrap_or_else(|_| "/app/fastembed_cache".to_string());

    let cache_dir = PathBuf::from(&cache_dir_str);
    println!("📁 Target cache directory: {}", cache_dir.display());

    std::fs::create_dir_all(&cache_dir)?;

    let options = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
        .with_cache_dir(cache_dir.clone())
        .with_show_download_progress(true);

    println!("⬇️  Downloading ONNX weights and tokenizer...");
    let mut model = TextEmbedding::try_new(options)?;

    // Run a warmup inference to verify model integrity and compile ONNX graph
    println!("🧪 Verifying model inference on dummy input...");
    let test_embedding = model.embed(vec!["warmup test prompt"], None)?;

    assert_eq!(test_embedding.len(), 1, "Expected 1 embedding vector");
    assert_eq!(
        test_embedding[0].len(),
        384,
        "Expected 384 dimensions for AllMiniLML6V2"
    );

    println!("============================================================");
    println!("✅ Model pre-download & verification successful!");
    println!("   Location: {}", cache_dir.display());
    println!("   Embedding dimension: 384");
    println!("============================================================");

    Ok(())
}
