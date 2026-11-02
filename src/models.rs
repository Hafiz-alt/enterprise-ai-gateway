// =============================================================================
// src/models.rs — OpenAI-compatible request / response types
// =============================================================================
//
// These structs mirror the subset of the OpenAI Chat Completions API that we
// need to extract cache keys, detect streaming mode, and parse SSE chunks.
// Fields we don't inspect are captured in `extra` via `#[serde(flatten)]` so
// the proxy forwards them to the upstream provider transparently.
//
// EXTENSION POINT: When you add Anthropic / Gemini support, create parallel
// structs (or an enum-dispatched provider model) in this module.
// =============================================================================

use serde::{Deserialize, Serialize};

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  INBOUND REQUEST                                                        ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// A (subset of the) OpenAI Chat Completion request.
///
/// We deserialize just enough to extract the prompt for cache lookup and the
/// `stream` flag for response-mode branching; every other field (`temperature`,
/// `top_p`, `tools`, …) is transparently captured in `extra` and forwarded
/// as-is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier (e.g. `"gpt-4o"`, `"gpt-3.5-turbo"`).
    pub model: String,

    /// The conversation messages.  We inspect only the **last user message**
    /// for cache-key extraction; modify this strategy when your semantic cache
    /// considers full conversation context.
    pub messages: Vec<Message>,

    /// When `true`, the upstream provider returns Server-Sent Events instead
    /// of a single JSON blob.  Drives the Phase 2 streaming branch in the
    /// handler.
    ///
    /// `skip_serializing_if` prevents us from sending `"stream": null` to the
    /// upstream when the client didn't include the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// All other fields forwarded verbatim to the upstream provider.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// One of `"system"`, `"user"`, `"assistant"`, `"tool"`.
    pub role: String,

    /// The text content of the message.
    /// NOTE: OpenAI also supports content as an array of content parts
    /// (for vision / multi-modal).  For Phase 1 we only handle the string
    /// form; extend this to an enum if you need multi-modal cache keys.
    pub content: Option<String>,

    /// Catch-all for `name`, `tool_calls`, `tool_call_id`, etc.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Request helpers
// ---------------------------------------------------------------------------

impl ChatCompletionRequest {
    /// Returns `true` if the client requested SSE streaming.
    pub fn is_streaming(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// Extract the cache key from the request.
    ///
    /// **Current strategy (Phase 1):** exact-match on the last `user` message.
    ///
    /// **INJECTION POINT — SEMANTIC CACHE:**
    /// Replace this method body with your embedding-based lookup:
    ///   1. Embed the prompt via your vector model.
    ///   2. Query your ANN index (HNSW / IVF-PQ / etc.).
    ///   3. Return the nearest-neighbor key if similarity ≥ threshold.
    ///
    /// The return type is `Option<String>` — return `None` if no meaningful
    /// prompt can be extracted (e.g., tool-only turn).
    pub fn cache_key(&self) -> Option<String> {
        // Walk backwards to find the most recent user message.
        self.messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .and_then(|m| m.content.clone())
            .filter(|c| !c.trim().is_empty())
    }
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║  STREAMING CHUNK TYPES  (Phase 2)                                       ║
// ║                                                                         ║
// ║  These types parse the `data:` payloads inside the SSE stream coming    ║
// ║  from OpenAI so we can extract content deltas for cache buffering.      ║
// ╚═══════════════════════════════════════════════════════════════════════════╝

/// A single SSE chunk from the OpenAI streaming response.
///
/// Example payload:
/// ```json
/// {
///   "id": "chatcmpl-abc123",
///   "object": "chat.completion.chunk",
///   "created": 1700000000,
///   "model": "gpt-4o",
///   "choices": [{
///     "index": 0,
///     "delta": { "content": "Hello" },
///     "finish_reason": null
///   }]
/// }
/// ```
#[derive(Debug, Deserialize)]
pub struct StreamChunk {
    /// Completion ID — consistent across all chunks in one response.
    #[serde(default)]
    pub id: Option<String>,

    /// Unix timestamp of when the completion was created.
    #[serde(default)]
    pub created: Option<u64>,

    /// The model that generated the chunk (e.g. `"gpt-4o"`).
    #[serde(default)]
    pub model: Option<String>,

    /// The choices array.  For single-generation requests this has one entry.
    #[serde(default)]
    pub choices: Vec<StreamChoice>,

    /// Token usage metrics (present in trailing stream chunks when requested).
    #[serde(default)]
    pub usage: Option<Usage>,
}

/// Token usage metadata returned by OpenAI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,

    #[serde(default)]
    pub completion_tokens: Option<u32>,

    #[serde(default)]
    pub total_tokens: Option<u32>,
}

/// A single choice inside a streaming chunk.
#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    /// Choice index (usually `0` for single generations).
    #[serde(default)]
    pub index: u32,

    /// The incremental delta — contains either a `role`, `content`, or both.
    pub delta: Delta,

    /// `null` while streaming; `"stop"` (or `"tool_calls"`, etc.) on the
    /// final chunk for this choice.
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Incremental content delta within a streaming choice.
#[derive(Debug, Deserialize)]
pub struct Delta {
    /// Present only on the very first chunk (value: `"assistant"`).
    #[serde(default)]
    pub role: Option<String>,

    /// The next fragment of the generated text.  `None` on the role-only
    /// first chunk and on the final finish chunk.
    #[serde(default)]
    pub content: Option<String>,
}

// ---------------------------------------------------------------------------
// Synthetic response construction
// ---------------------------------------------------------------------------

/// Build a standard (non-streaming) `ChatCompletion` JSON string from the
/// pieces we accumulated while intercepting a stream.
///
/// This is what we store in the cache — a canonical non-streaming format —
/// so the same cached entry works for both streaming and non-streaming
/// subsequent requests.
pub fn build_synthetic_response(
    id: &str,
    model: &str,
    content: &str,
    created: u64,
    usage: Option<&Usage>,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": usage
    })
    .to_string()
}

/// Parse a cached non-streaming response to extract the fields we need
/// for streaming simulation.
///
/// Returns `(id, model, content, created)` or `None` if the JSON is
/// malformed or missing expected fields.
pub fn parse_cached_response(json: &str) -> Option<(String, String, String, u64)> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let id = v.get("id")?.as_str()?.to_string();
    let model = v.get("model")?.as_str()?.to_string();
    let content = v
        .get("choices")?
        .get(0)?
        .get("message")?
        .get("content")?
        .as_str()?
        .to_string();
    let created = v.get("created").and_then(|c| c.as_u64()).unwrap_or(0);
    Some((id, model, content, created))
}
