// =============================================================================
// src/handler.rs — The core proxy handler for /v1/chat/completions
// =============================================================================
//
// Instrumented with OpenTelemetry for Langfuse tracing:
//   - Root Span: `chat_completions`
//   - Attributes:
//       • `cache.status`: "HIT" | "MISS"
//       • `cache.similarity_score`: Cosine similarity score on hit
//       • `llm.usage.prompt_tokens`: Input tokens
//       • `llm.usage.completion_tokens`: Output tokens
//       • `llm.model`: Model requested
//       • `stream`: Whether streaming was requested
//
// =============================================================================

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures_util::StreamExt;
use reqwest::Client;

use crate::cache::LlmCache;
use crate::error::AppError;
use crate::models::{
    build_synthetic_response, parse_cached_response, ChatCompletionRequest, StreamChunk, Usage,
};

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  SHARED APPLICATION STATE                                               ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// Everything the handler needs, wrapped in a cheaply-cloneable struct.
/// Axum requires `State<T>` to be `Clone`.
#[derive(Clone)]
pub struct AppState {
    /// The semantic LLM response cache.
    pub cache: LlmCache,

    /// A persistent, connection-pooled HTTP client for upstream calls.
    pub http_client: Client,

    /// The upstream provider's base URL.
    pub upstream_url: String,
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  MAIN ENTRY POINT                                                       ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// POST /v1/chat/completions
///
/// Accepts an OpenAI-formatted chat completion request, detects whether the
/// client wants streaming or blocking, checks the cache, and either returns
/// a cached response or proxies to the upstream provider.
#[tracing::instrument(
    name = "chat_completions",
    skip_all,
    fields(
        cache.status = tracing::field::Empty,
        cache.similarity_score = tracing::field::Empty,
        llm.usage.prompt_tokens = tracing::field::Empty,
        llm.usage.completion_tokens = tracing::field::Empty,
        llm.model = %request.model,
        stream = request.is_streaming(),
    )
)]
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, AppError> {
    // ── 1. Extract the cache key ──────────────────────────────────────
    let cache_key = request.cache_key().ok_or_else(|| {
        AppError::BadRequest(
            "No user message found in the request to use as cache key".into(),
        )
    })?;

    // ── 2. Branch on streaming mode ───────────────────────────────────
    if request.is_streaming() {
        tracing::info!("streaming request detected");
        handle_streaming(state, request, cache_key).await
    } else {
        handle_non_streaming(state, request, cache_key).await
    }
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  NON-STREAMING PATH                                                      ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// Handle a standard (non-streaming) chat completion request.
///
/// Cache hit  → return the cached JSON immediately, record telemetry.
/// Cache miss → forward to upstream, parse tokens, cache response, return.
async fn handle_non_streaming(
    state: AppState,
    request: ChatCompletionRequest,
    cache_key: String,
) -> Result<Response, AppError> {
    let span = tracing::Span::current();

    // ── Cache lookup ──────────────────────────────────────────────────
    if let Some(hit) = state.cache.get(&cache_key).await {
        tracing::info!("serving non-streaming response from cache");

        // Record cache telemetry
        span.record("cache.status", "HIT");
        span.record("cache.similarity_score", hit.similarity_score);

        // Record cached token counts if present in payload
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&hit.payload) {
            if let Some(usage) = v.get("usage") {
                if let Some(pt) = usage.get("prompt_tokens").and_then(|x| x.as_u64()) {
                    span.record("llm.usage.prompt_tokens", pt);
                }
                if let Some(ct) = usage.get("completion_tokens").and_then(|x| x.as_u64()) {
                    span.record("llm.usage.completion_tokens", ct);
                }
            }
        }

        return Ok((
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            [(header::HeaderName::from_static("x-cache"), "HIT")],
            hit.payload.as_str().to_owned(),
        )
            .into_response());
    }

    // ── Cache miss → forward to upstream ──────────────────────────────
    span.record("cache.status", "MISS");
    tracing::info!("cache miss — forwarding non-streaming to upstream");

    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AppError::MissingApiKey)?;
    let request_body = serde_json::to_vec(&request)?;

    let upstream_response = state
        .http_client
        .post(&state.upstream_url)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
        .body(request_body)
        .send()
        .await?;

    let upstream_status = StatusCode::from_u16(upstream_response.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);

    let response_body = upstream_response.text().await?;

    if !upstream_status.is_success() {
        return Err(AppError::UpstreamHttp {
            status: upstream_status,
            body: response_body,
        });
    }

    // ── Record token usage metrics from upstream JSON ─────────────────
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&response_body) {
        if let Some(usage) = val.get("usage") {
            if let Some(pt) = usage.get("prompt_tokens").and_then(|x| x.as_u64()) {
                span.record("llm.usage.prompt_tokens", pt);
            }
            if let Some(ct) = usage.get("completion_tokens").and_then(|x| x.as_u64()) {
                span.record("llm.usage.completion_tokens", ct);
            }
        }
    }

    // Cache + return
    state.cache.set(cache_key, response_body.clone()).await;

    Ok((
        upstream_status,
        [(header::CONTENT_TYPE, "application/json")],
        [(header::HeaderName::from_static("x-cache"), "MISS")],
        response_body,
    )
        .into_response())
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  STREAMING PATH                                                         ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// Handle a streaming (`stream: true`) chat completion request.
///
/// Cache hit  → simulate an SSE stream from the cached non-streaming JSON.
/// Cache miss → proxy the upstream SSE stream, tee to client + buffer, cache.
async fn handle_streaming(
    state: AppState,
    request: ChatCompletionRequest,
    cache_key: String,
) -> Result<Response, AppError> {
    let span = tracing::Span::current();

    // ── Cache lookup ──────────────────────────────────────────────────
    if let Some(hit) = state.cache.get(&cache_key).await {
        tracing::info!("serving streaming response from cache (simulated SSE)");

        span.record("cache.status", "HIT");
        span.record("cache.similarity_score", hit.similarity_score);

        // Record cached token counts if present in payload
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&hit.payload) {
            if let Some(usage) = v.get("usage") {
                if let Some(pt) = usage.get("prompt_tokens").and_then(|x| x.as_u64()) {
                    span.record("llm.usage.prompt_tokens", pt);
                }
                if let Some(ct) = usage.get("completion_tokens").and_then(|x| x.as_u64()) {
                    span.record("llm.usage.completion_tokens", ct);
                }
            }
        }

        return Ok(build_cached_sse_response(hit.payload));
    }

    // ── Cache miss → proxy upstream SSE ───────────────────────────────
    span.record("cache.status", "MISS");
    tracing::info!("cache miss — forwarding streaming to upstream");
    build_upstream_sse_response(state, request, cache_key, span).await
}

// ---------------------------------------------------------------------------
// Cache HIT → Simulated SSE stream
// ---------------------------------------------------------------------------

/// Build an SSE response that replays a cached non-streaming response as a
/// series of streaming chunks.
fn build_cached_sse_response(cached: Arc<String>) -> Response {
    let stream = async_stream::stream! {
        let (id, model, content, created) = match parse_cached_response(&cached) {
            Some(fields) => fields,
            None => {
                tracing::warn!("could not parse cached response for SSE simulation");
                yield Ok::<Event, Infallible>(Event::default().data(cached.as_str().to_owned()));
                yield Ok::<Event, Infallible>(Event::default().data("[DONE]".to_string()));
                return;
            }
        };

        // 1. Role chunk
        let role_chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "role": "assistant" },
                "finish_reason": serde_json::Value::Null
            }]
        });
        yield Ok::<Event, Infallible>(Event::default().data(role_chunk.to_string()));

        // 2. Content chunks (word-by-word with delay)
        let words: Vec<&str> = content.split_inclusive(char::is_whitespace).collect();

        for word in &words {
            tokio::time::sleep(Duration::from_millis(10)).await;

            let chunk = serde_json::json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": { "content": word },
                    "finish_reason": serde_json::Value::Null
                }]
            });
            yield Ok::<Event, Infallible>(Event::default().data(chunk.to_string()));
        }

        // 3. Finish chunk
        let finish_chunk = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        });
        yield Ok::<Event, Infallible>(Event::default().data(finish_chunk.to_string()));

        // 4. [DONE] sentinel
        yield Ok::<Event, Infallible>(Event::default().data("[DONE]".to_string()));

        tracing::debug!("finished simulated SSE stream from cache");
    };

    let mut response = Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response();

    response.headers_mut().insert(
        header::HeaderName::from_static("x-cache"),
        header::HeaderValue::from_static("HIT"),
    );

    response
}

// ---------------------------------------------------------------------------
// Cache MISS → Upstream SSE proxy with token extraction & buffering
// ---------------------------------------------------------------------------

/// Forward the request to the upstream provider as an SSE stream, tee
/// each chunk to the client, capture tokens onto the parent span, and cache.
async fn build_upstream_sse_response(
    state: AppState,
    request: ChatCompletionRequest,
    cache_key: String,
    parent_span: tracing::Span,
) -> Result<Response, AppError> {
    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| AppError::MissingApiKey)?;
    let request_body = serde_json::to_vec(&request)?;

    let upstream_response = state
        .http_client
        .post(&state.upstream_url)
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {api_key}"))
        .body(request_body)
        .send()
        .await?;

    let upstream_status = StatusCode::from_u16(upstream_response.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);

    if !upstream_status.is_success() {
        let body = upstream_response.text().await?;
        return Err(AppError::UpstreamHttp {
            status: upstream_status,
            body,
        });
    }

    let byte_stream = upstream_response.bytes_stream();
    let cache = state.cache.clone();

    let event_stream = async_stream::stream! {
        tokio::pin!(byte_stream);

        let mut line_buf = String::new();
        let mut content_buf = String::new();
        let mut chunk_id = String::from("chatcmpl-unknown");
        let mut chunk_model = String::from("unknown");
        let mut chunk_created: u64 = 0;
        let mut captured_usage: Option<Usage> = None;

        while let Some(chunk_result) = byte_stream.next().await {
            let bytes = match chunk_result {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!(error = %e, "upstream byte stream error");
                    break;
                }
            };

            line_buf.push_str(&String::from_utf8_lossy(&bytes));

            while let Some(boundary) = line_buf.find("\n\n") {
                let event_text = line_buf[..boundary].to_string();
                line_buf = line_buf[boundary + 2..].to_string();

                for line in event_text.lines() {
                    let data = match line.strip_prefix("data: ")
                        .or_else(|| line.strip_prefix("data:"))
                    {
                        Some(d) => d.trim(),
                        None => continue,
                    };

                    if data == "[DONE]" {
                        yield Ok::<Event, Infallible>(
                            Event::default().data("[DONE]".to_string())
                        );
                        continue;
                    }

                    match serde_json::from_str::<StreamChunk>(data) {
                        Ok(chunk) => {
                            if let Some(ref id) = chunk.id {
                                if chunk_id == "chatcmpl-unknown" {
                                    chunk_id = id.clone();
                                }
                            }
                            if let Some(ref model) = chunk.model {
                                if chunk_model == "unknown" {
                                    chunk_model = model.clone();
                                }
                            }
                            if let Some(created) = chunk.created {
                                if chunk_created == 0 {
                                    chunk_created = created;
                                }
                            }

                            // Capture usage if present in the chunk
                            if let Some(usage) = chunk.usage {
                                captured_usage = Some(usage);
                            }

                            // Accumulate content deltas
                            if let Some(delta_content) = chunk
                                .choices
                                .first()
                                .and_then(|c| c.delta.content.as_ref())
                            {
                                content_buf.push_str(delta_content);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                data_prefix = &data[..data.len().min(120)],
                                "failed to parse SSE chunk JSON"
                            );
                        }
                    }

                    yield Ok::<Event, Infallible>(
                        Event::default().data(data.to_string())
                    );
                }
            }
        }

        // ── Stream complete: record token metrics on the parent span ───
        if let Some(ref usage) = captured_usage {
            if let Some(pt) = usage.prompt_tokens {
                parent_span.record("llm.usage.prompt_tokens", pt as u64);
            }
            if let Some(ct) = usage.completion_tokens {
                parent_span.record("llm.usage.completion_tokens", ct as u64);
            }
        } else if !content_buf.is_empty() {
            // If the provider didn't include usage in the stream, approximate tokens
            let est_tokens = (content_buf.split_whitespace().count() as f32 * 1.3) as u64;
            parent_span.record("llm.usage.completion_tokens", est_tokens);
        }

        // Cache the full response if content was received
        if !content_buf.is_empty() {
            let synthetic = build_synthetic_response(
                &chunk_id,
                &chunk_model,
                &content_buf,
                chunk_created,
                captured_usage.as_ref(),
            );

            cache.set(cache_key, synthetic).await;

            tracing::info!(
                content_len = content_buf.len(),
                chunk_id = %chunk_id,
                "cached streamed response as synthetic non-streaming JSON"
            );
        } else {
            tracing::warn!("upstream stream ended with no content — skipping cache");
        }
    };

    let mut response = Sse::new(event_stream)
        .keep_alive(KeepAlive::default())
        .into_response();

    response.headers_mut().insert(
        header::HeaderName::from_static("x-cache"),
        header::HeaderValue::from_static("MISS"),
    );

    Ok(response)
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  HEALTH CHECK                                                           ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// GET /health
///
/// Returns 200 with basic cache and vector stats.
pub async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "cache_entries": state.cache.entry_count(),
        "vector_entries": state.cache.vector_count(),
        "similarity_threshold": state.cache.similarity_threshold(),
    }))
}
