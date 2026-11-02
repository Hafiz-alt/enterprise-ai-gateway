// =============================================================================
// src/cache.rs — High-performance in-memory semantic LLM response cache
// =============================================================================
//
// Dual-store semantic caching architecture:
//
//   1. Local Embedding Engine:
//      - Backed by `fastembed::TextEmbedding` (AllMiniLML6V2, 384-dimensional).
//      - Runs in-process via ONNX Runtime without network round-trips.
//      - Offloaded to `tokio::task::spawn_blocking` to protect the Tokio reactor.
//
//   2. Vector Index (`Arc<RwLock<Vec<(String, Vec<f32>)>>>`):
//      - Stores `(cache_key, embedding)` tuples for fast in-memory linear scans.
//      - Synchronized with Moka evictions via an `eviction_listener`.
//
//   3. Response Store (`moka::future::Cache<String, Arc<String>>`):
//      - Handles TTL expiration, maximum capacity limits, and LFU eviction.
//
// Cosine Similarity:
//   similarity(A, B) = (A · B) / (||A|| * ||B||)
//   If similarity >= `similarity_threshold` (e.g. 0.92), returns a cache HIT.
//
// =============================================================================

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use moka::future::Cache;
use moka::notification::RemovalCause;

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  OPTIMIZED COSINE SIMILARITY                                             ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// Compute cosine similarity between two vector slices.
///
/// Single-pass calculation computing dot product and both Euclidean norms
/// simultaneously to maximize cache locality and allow compiler SIMD vectorization.
///
/// Returns a value in [-1.0, 1.0].
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_prod = (norm_a * norm_b).sqrt();
    if norm_prod > 0.0 {
        dot / norm_prod
    } else {
        0.0
    }
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  SEMANTIC LLM CACHE                                                     ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// Thread-safe in-memory semantic LLM response cache.
///
/// Cheaply cloneable across Axum handlers via `State` extractor (all internal
/// structures are wrapped in `Arc`).
#[derive(Clone)]
pub struct LlmCache {
    /// In-memory response cache for TTL / eviction management.
    /// Key:   raw user prompt (or canonical key)
    /// Value: serialized JSON response body (Arc<String>)
    inner: Cache<String, Arc<String>>,

    /// In-memory linear vector index: (cache_key, embedding).
    /// Guarded by `RwLock` for high-throughput concurrent reads during scans.
    vectors: Arc<RwLock<Vec<(String, Vec<f32>)>>>,

    /// Fast local embedding model. `fastembed::TextEmbedding::embed` takes
    /// `&mut self`, so it is guarded by `Arc<Mutex<...>>` and accessed only
    /// inside `tokio::task::spawn_blocking`.
    embedding_model: Arc<Mutex<TextEmbedding>>,

    /// Minimum cosine similarity score (0.0 to 1.0) required to claim a hit.
    similarity_threshold: f32,
}

/// Metadata and payload returned on a semantic cache hit.
#[derive(Debug, Clone)]
pub struct CacheHit {
    /// Cached JSON response payload.
    pub payload: Arc<String>,
    /// Cosine similarity score that triggered the hit (in [-1.0, 1.0]).
    pub similarity_score: f32,
}

impl LlmCache {
    /// Initialize the semantic cache with capacity, TTL, and similarity threshold.
    ///
    /// Loads the `AllMiniLML6V2` embedding model locally on startup.
    ///
    /// # Arguments
    /// * `max_capacity` — Maximum number of entries before Moka initiates LFU eviction.
    /// * `ttl` — Time-to-live per entry.
    /// * `similarity_threshold` — Minimum cosine similarity score (e.g. 0.92) to match.
    pub fn new(
        max_capacity: u64,
        ttl: Duration,
        similarity_threshold: f32,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        tracing::info!("initializing local AllMiniLML6V2 embedding engine...");

        // Load the lightweight, high-performance AllMiniLML6V2 model (384 dims).
        // Respect FASTEMBED_CACHE_PATH or FASTEMBED_CACHE_DIR so Docker/airgapped
        // deployments load models directly from local disk without outbound internet.
        let cache_dir = std::env::var("FASTEMBED_CACHE_PATH")
            .or_else(|_| std::env::var("FASTEMBED_CACHE_DIR"))
            .ok()
            .map(PathBuf::from);

        let mut init_options = TextInitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(true);

        if let Some(ref path) = cache_dir {
            tracing::info!(cache_path = %path.display(), "using pre-baked fastembed cache directory");
            init_options = init_options.with_cache_dir(path.clone());
        }

        let model = TextEmbedding::try_new(init_options)?;

        let vectors = Arc::new(RwLock::new(Vec::new()));
        let vectors_for_eviction = Arc::clone(&vectors);

        // Hook Moka's eviction listener to keep the vector store in sync when entries expire.
        let inner = Cache::builder()
            .max_capacity(max_capacity)
            .time_to_live(ttl)
            .eviction_listener(
                move |key: Arc<String>, _val: Arc<String>, cause: RemovalCause| {
                    if cause != RemovalCause::Replaced {
                        if let Ok(mut vecs) = vectors_for_eviction.write() {
                            let before_len = vecs.len();
                            vecs.retain(|(k, _)| k != &*key);
                            if vecs.len() < before_len {
                                tracing::debug!(
                                    evicted_key = %key,
                                    ?cause,
                                    "pruned vector index following Moka eviction"
                                );
                            }
                        }
                    }
                },
            )
            .build();

        tracing::info!(
            similarity_threshold,
            max_capacity,
            ttl_secs = ttl.as_secs(),
            "semantic LlmCache ready"
        );

        Ok(Self {
            inner,
            vectors,
            embedding_model: Arc::new(Mutex::new(model)),
            similarity_threshold,
        })
    }

    /// Look up a cached response by semantic similarity.
    ///
    /// Returns `Some(CacheHit)` if any cached prompt has cosine
    /// similarity >= `similarity_threshold`, otherwise `None`.
    pub async fn get(&self, prompt: &str) -> Option<CacheHit> {
        // Fast-path: If the vector index has no entries, avoid ONNX inference entirely.
        {
            let vecs = self.vectors.read().ok()?;
            if vecs.is_empty() {
                tracing::debug!("semantic cache empty — immediate miss");
                return None;
            }
        }

        let model = Arc::clone(&self.embedding_model);
        let vectors = Arc::clone(&self.vectors);
        let prompt_text = prompt.to_string();
        let threshold = self.similarity_threshold;

        // Offload CPU-bound embedding generation and vector search to Tokio's blocking pool.
        let match_result = tokio::task::spawn_blocking(move || -> Option<(String, f32)> {
            // 1. Generate dense embedding vector for the incoming prompt
            // Instrument with vector_generation_span to monitor ONNX model inference latency
            let query_vec = {
                let _span = tracing::info_span!("vector_generation_span").entered();
                let mut model_guard = model.lock().ok()?;
                let mut embeddings = model_guard.embed(vec![prompt_text], None).ok()?;
                embeddings.pop()?
            };

            // 2. Linear scan over indexed vectors for highest cosine similarity
            // Instrument with similarity_scan_span to monitor vector index search latency
            let best_match = {
                let _span = tracing::info_span!("similarity_scan_span").entered();
                let vecs_guard = vectors.read().ok()?;
                let mut best: Option<(String, f32)> = None;

                for (cached_key, cached_vec) in vecs_guard.iter() {
                    let similarity = cosine_similarity(&query_vec, cached_vec);
                    if similarity >= threshold {
                        if let Some((_, best_sim)) = best {
                            if similarity > best_sim {
                                best = Some((cached_key.clone(), similarity));
                            }
                        } else {
                            best = Some((cached_key.clone(), similarity));
                        }
                    }
                }

                best
            };

            best_match
        })
        .await
        .ok()?;

        // If a match exceeded threshold, retrieve the cached response payload from Moka.
        if let Some((matched_key, score)) = match_result {
            if let Some(cached_response) = self.inner.get(&matched_key).await {
                tracing::info!(
                    similarity = score,
                    threshold = self.similarity_threshold,
                    matched_key_prefix = &matched_key[..matched_key.len().min(60)],
                    "semantic cache HIT"
                );
                return Some(CacheHit {
                    payload: cached_response,
                    similarity_score: score,
                });
            } else {
                // Key expired from Moka or was removed
                tracing::debug!(
                    similarity = score,
                    matched_key_prefix = &matched_key[..matched_key.len().min(60)],
                    "semantic vector matched but payload was evicted from Moka"
                );
            }
        } else {
            tracing::debug!(
                threshold = self.similarity_threshold,
                prompt_prefix = &prompt[..prompt.len().min(60)],
                "semantic cache MISS (similarity below threshold)"
            );
        }

        None
    }

    /// Insert a prompt and its response into both the vector index and Moka.
    ///
    /// Computes the prompt's embedding inside `spawn_blocking` to avoid
    /// stalling the async worker threads.
    pub async fn set(&self, prompt: String, response: String) {
        let model = Arc::clone(&self.embedding_model);
        let prompt_for_embed = prompt.clone();

        // Generate embedding off-thread
        let embed_task = tokio::task::spawn_blocking(move || -> Result<Vec<f32>, String> {
            let _span = tracing::info_span!("vector_generation_span").entered();
            let mut model_guard = model
                .lock()
                .map_err(|e| format!("model lock error: {e}"))?;
            let mut embeddings = model_guard
                .embed(vec![prompt_for_embed], None)
                .map_err(|e| format!("embed error: {e}"))?;
            embeddings
                .pop()
                .ok_or_else(|| "no embedding returned by fastembed".to_string())
        })
        .await;

        match embed_task {
            Ok(Ok(embedding)) => {
                // 1. Upsert vector in the linear index
                {
                    if let Ok(mut vecs_guard) = self.vectors.write() {
                        if let Some(existing) = vecs_guard.iter_mut().find(|(k, _)| k == &prompt) {
                            existing.1 = embedding;
                        } else {
                            vecs_guard.push((prompt.clone(), embedding));
                        }
                    }
                }

                // 2. Insert into Moka (manages TTL and capacity)
                self.inner.insert(prompt.clone(), Arc::new(response)).await;

                tracing::debug!(
                    prompt_prefix = &prompt[..prompt.len().min(60)],
                    vector_count = self.vector_count(),
                    "semantic cache SET"
                );
            }
            Ok(Err(err)) => {
                tracing::error!(error = %err, "failed to compute embedding for cache set");
            }
            Err(join_err) => {
                tracing::error!(error = %join_err, "spawn_blocking panicked during embedding");
            }
        }
    }

    /// Return the count of payloads currently in Moka.
    pub fn entry_count(&self) -> u64 {
        self.inner.entry_count()
    }

    /// Return the count of indexed vectors.
    pub fn vector_count(&self) -> usize {
        self.vectors.read().map(|v| v.len()).unwrap_or(0)
    }

    /// Return the configured similarity threshold.
    pub fn similarity_threshold(&self) -> f32 {
        self.similarity_threshold
    }
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  UNIT TESTS                                                             ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let v1 = vec![1.0, 0.0];
        let v2 = vec![-1.0, 0.0];
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_empty_or_mismatched() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_cosine_similarity_scaled_vectors() {
        let v1 = vec![1.0, 2.0, 3.0];
        let v2 = vec![2.0, 4.0, 6.0]; // same direction, double magnitude
        let sim = cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-5);
    }
}
