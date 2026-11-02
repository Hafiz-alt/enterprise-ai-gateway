// =============================================================================
// src/main.rs — Entry point for the AI Gateway
// =============================================================================
//
// Wires together the router, state, and tracing infrastructure.
//
// Usage:
//   OPENAI_API_KEY=sk-... cargo run
//
// Then hit the gateway exactly as you would OpenAI:
//   curl http://localhost:8080/v1/chat/completions \
//     -H "Content-Type: application/json" \
//     -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Hello!"}]}'
//
// =============================================================================

mod cache;
mod error;
mod handler;
mod models;
mod telemetry;

use std::time::Duration;

use axum::{routing, Router};
use reqwest::Client;
use tower_http::trace::TraceLayer;

use cache::LlmCache;
use handler::AppState;

// ---------------------------------------------------------------------------
// Configuration — read from env vars with sensible defaults.
// In production, pull these from your config management system (Consul, etc.)
// ---------------------------------------------------------------------------

/// Upstream URL for the LLM provider.
/// Override with `UPSTREAM_URL` env var to target Azure OpenAI, vLLM, etc.
fn upstream_url() -> String {
    std::env::var("UPSTREAM_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1/chat/completions".to_string())
}

/// Port the gateway listens on.
fn listen_port() -> u16 {
    std::env::var("GATEWAY_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080)
}

/// Maximum cache capacity (number of entries).
fn cache_max_capacity() -> u64 {
    std::env::var("CACHE_MAX_CAPACITY")
        .ok()
        .and_then(|c| c.parse().ok())
        .unwrap_or(10_000)
}

/// Cache entry time-to-live in seconds.
fn cache_ttl_secs() -> u64 {
    std::env::var("CACHE_TTL_SECS")
        .ok()
        .and_then(|t| t.parse().ok())
        .unwrap_or(3600)
}

/// Cosine similarity threshold for semantic cache hits (0.0 to 1.0).
fn similarity_threshold() -> f32 {
    std::env::var("SIMILARITY_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.92)
}

// ---------------------------------------------------------------------------
// Graceful shutdown signal listener
// ---------------------------------------------------------------------------

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Ctrl+C received — initiating graceful shutdown");
        },
        _ = terminate => {
            tracing::info!("SIGTERM received — initiating graceful shutdown");
        },
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // ── 1. Initialize OpenTelemetry & Langfuse OTLP Tracing ───────────
    let _tracer_provider = telemetry::init_telemetry()
        .expect("failed to initialize OpenTelemetry tracing");

    // ── 2. Build shared application state ─────────────────────────────
    let cache = LlmCache::new(
        cache_max_capacity(),
        Duration::from_secs(cache_ttl_secs()),
        similarity_threshold(),
    )
    .expect("failed to initialize semantic embedding model");

    let http_client = Client::builder()
        // Aggressive timeouts — if OpenAI is slow, fail fast so the
        // client can retry or fall back.
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(120))
        // Connection pool: keep warm connections to upstream.
        .pool_max_idle_per_host(20)
        .build()
        .expect("failed to build reqwest Client");

    let state = AppState {
        cache,
        http_client,
        upstream_url: upstream_url(),
    };

    // ── 3. Build the Axum router ──────────────────────────────────────
    let app = Router::new()
        // Core proxy route — mirrors the OpenAI endpoint path.
        .route(
            "/v1/chat/completions",
            routing::post(handler::chat_completions),
        )
        // Health check for k8s / ALB probes.
        .route("/health", routing::get(handler::health_check))
        // Attach shared state.
        .with_state(state)
        // Request / response tracing via tower-http.
        .layer(TraceLayer::new_for_http());

    // ── 4. Bind & serve with graceful shutdown ────────────────────────
    let port = listen_port();
    let addr = format!("0.0.0.0:{port}");

    tracing::info!(%addr, "🚀 AI Gateway starting");
    tracing::info!(upstream = %upstream_url(), "proxying to upstream LLM provider");
    tracing::info!(
        capacity = cache_max_capacity(),
        ttl_secs = cache_ttl_secs(),
        threshold = similarity_threshold(),
        "semantic cache initialized"
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind TCP listener");

    // Serve with graceful shutdown so all in-flight requests finish cleanly
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server exited unexpectedly");

    // ── 5. Flush remaining OTel spans before process termination ──────
    telemetry::shutdown_telemetry();
}
