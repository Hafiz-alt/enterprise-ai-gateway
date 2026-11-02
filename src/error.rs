// =============================================================================
// src/error.rs — Centralized error type for the gateway
// =============================================================================
//
// Every fallible path in the handler returns `AppError`.  Axum converts it to
// an HTTP response via the `IntoResponse` impl below, so callers never need
// raw `.unwrap()` — they just `?`-propagate.
//
// EXTENSION POINT: Add new variants here when you introduce database-backed
// caching, rate limiting, or auth middleware.
// =============================================================================

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

/// Unified application error that maps cleanly to HTTP status codes.
#[derive(Debug)]
pub enum AppError {
    /// The `OPENAI_API_KEY` environment variable is missing or empty.
    MissingApiKey,

    /// The inbound request body couldn't be parsed as a valid
    /// `ChatCompletionRequest`, or it contained no extractable user prompt.
    BadRequest(String),

    /// The upstream LLM provider returned a non-2xx response.
    /// Carries the status code and the raw body so we can relay diagnostics.
    UpstreamHttp {
        status: StatusCode,
        body: String,
    },

    /// Network-level failure reaching the upstream provider (DNS, timeout, etc.).
    UpstreamNetwork(reqwest::Error),

    /// JSON serialization / deserialization failed.
    Serialization(serde_json::Error),
}

// ---------------------------------------------------------------------------
// Trait impls
// ---------------------------------------------------------------------------

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingApiKey => {
                write!(f, "OPENAI_API_KEY is not set in the environment")
            }
            Self::BadRequest(msg) => write!(f, "Bad request: {msg}"),
            Self::UpstreamHttp { status, body } => {
                write!(f, "Upstream returned {status}: {body}")
            }
            Self::UpstreamNetwork(e) => write!(f, "Upstream network error: {e}"),
            Self::Serialization(e) => write!(f, "JSON error: {e}"),
        }
    }
}

impl std::error::Error for AppError {}

// Convert `reqwest::Error` → `AppError` so `?` works seamlessly.
impl From<reqwest::Error> for AppError {
    fn from(err: reqwest::Error) -> Self {
        Self::UpstreamNetwork(err)
    }
}

// Convert `serde_json::Error` → `AppError`.
impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization(err)
    }
}

// ---------------------------------------------------------------------------
// Axum integration — every `AppError` becomes a well-formed JSON response
// ---------------------------------------------------------------------------

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Self::MissingApiKey => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Server misconfiguration: missing API key".to_string(),
            ),
            Self::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            Self::UpstreamHttp { status, body } => (*status, body.clone()),
            Self::UpstreamNetwork(_) => (
                StatusCode::BAD_GATEWAY,
                "Failed to reach upstream LLM provider".to_string(),
            ),
            Self::Serialization(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "Invalid JSON payload".to_string(),
            ),
        };

        // Log the full error for observability; return a safe message to client.
        tracing::error!(error = %self, "request failed");

        let body = serde_json::json!({
            "error": {
                "message": message,
                "type": "gateway_error",
            }
        });

        (status, axum::Json(body)).into_response()
    }
}
