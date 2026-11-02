// =============================================================================
// src/telemetry.rs — OpenTelemetry & Langfuse OTLP Tracing Integration
// =============================================================================
//
// Tracing Architecture:
//
//   Application Spans (tracing)
//           │
//           ▼
//   tracing-opentelemetry layer
//           │
//           ▼
//   OpenTelemetry SdkTracerProvider (Batch Span Processor)
//           │ (asynchronous background flush)
//           ▼
//   opentelemetry-otlp HTTP Exporter (Binary Protobuf over HTTP)
//           │  Headers: Authorization: Basic <base64(pk:sk)>
//           │           x-langfuse-ingestion-version: 4
//           ▼
//   Langfuse OTLP Endpoint (e.g. https://cloud.langfuse.com/api/public/otel)
//
// =============================================================================

use std::collections::HashMap;
use std::time::Duration;

use base64::Engine;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::{Protocol, SpanExporter, WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::trace::{BatchConfigBuilder, SdkTracerProvider};
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// OTLP HTTP endpoint fallback for Langfuse.
const DEFAULT_LANGFUSE_ENDPOINT: &str = "https://cloud.langfuse.com/api/public/otel";

/// Initialize OpenTelemetry tracing with Langfuse OTLP export and console logging.
///
/// Sets up:
/// 1. An OTLP HTTP span exporter pointed at Langfuse (or custom OTLP backend).
/// 2. Basic Auth headers derived from `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.
/// 3. An asynchronous batch span processor to avoid blocking request paths.
/// 4. A `tracing-subscriber` Registry combining EnvFilter, fmt (stdout), and OTel layer.
pub fn init_telemetry() -> Result<SdkTracerProvider, Box<dyn std::error::Error + Send + Sync>> {
    // ── 1. Resolve OTLP endpoint ──────────────────────────────────────
    let base_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| DEFAULT_LANGFUSE_ENDPOINT.to_string());

    // Ensure trace path is correct for HTTP exporter
    let endpoint = if !base_endpoint.ends_with("/v1/traces") {
        format!("{}/v1/traces", base_endpoint.trim_end_matches('/'))
    } else {
        base_endpoint
    };

    // ── 2. Construct Langfuse / OTLP HTTP Headers ─────────────────────
    let mut headers = HashMap::new();

    // Check for explicit Langfuse API credentials
    let public_key = std::env::var("LANGFUSE_PUBLIC_KEY").ok();
    let secret_key = std::env::var("LANGFUSE_SECRET_KEY").ok();

    if let (Some(pk), Some(sk)) = (public_key, secret_key) {
        let auth_raw = format!("{pk}:{sk}");
        let auth_b64 = base64::engine::general_purpose::STANDARD.encode(auth_raw.as_bytes());
        headers.insert("Authorization".to_string(), format!("Basic {auth_b64}"));
        headers.insert("x-langfuse-ingestion-version".to_string(), "4".to_string());
        tracing::info!("Langfuse authentication headers configured");
    } else {
        tracing::warn!(
            "LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set; exporting without Basic Auth"
        );
    }

    // Also parse any standard OTEL_EXPORTER_OTLP_HEADERS (key1=val1,key2=val2)
    if let Ok(env_headers) = std::env::var("OTEL_EXPORTER_OTLP_HEADERS") {
        for pair in env_headers.split(',') {
            if let Some((k, v)) = pair.split_once('=') {
                headers.insert(k.trim().to_string(), v.trim().to_string());
            }
        }
    }

    // ── 3. Configure the OTLP HTTP Span Exporter ──────────────────────
    let exporter = SpanExporter::builder()
        .with_http()
        .with_protocol(Protocol::HttpBinary)
        .with_endpoint(&endpoint)
        .with_headers(headers)
        .with_timeout(Duration::from_secs(5))
        .build()?;

    // ── 4. Build Resource metadata ────────────────────────────────────
    let resource = Resource::new(vec![
        KeyValue::new("service.name", "ai-gateway"),
        KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
        KeyValue::new("telemetry.sdk.language", "rust"),
    ]);

    // ── 5. Build SdkTracerProvider with Batch Processor ───────────────
    // The batch processor accumulates spans and flushes them asynchronously
    // in the background, ensuring zero latency penalty on request handlers.
    let batch_config = BatchConfigBuilder::default()
        .with_max_queue_size(2048)
        .with_max_export_batch_size(512)
        .with_scheduled_delay(Duration::from_millis(500))
        .build();

    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(resource)
        .build();

    // Set globally so child tasks & libraries can acquire tracers
    global::set_tracer_provider(tracer_provider.clone());

    // ── 6. Compose tracing-subscriber layers ──────────────────────────
    let tracer = tracer_provider.tracer("ai-gateway");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        "ai_gateway=info,tower_http=debug,opentelemetry=warn,warn".into()
    });

    Registry::default()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

    tracing::info!(%endpoint, "OpenTelemetry & Langfuse exporter initialized successfully");

    Ok(tracer_provider)
}

/// Flush pending batches and shut down the global OpenTelemetry tracer provider.
pub fn shutdown_telemetry() {
    tracing::info!("flushing pending spans to Langfuse...");
    global::shutdown_tracer_provider();
    tracing::info!("OpenTelemetry tracer provider shutdown complete");
}
