<div align="center">

# ⚡ Enterprise AI Gateway (Rust)

**Ultra-low-latency, asynchronous reverse proxy for LLMs featuring in-memory Semantic Vector Caching, Server-Sent Events (SSE) streaming, and OpenTelemetry distributed tracing to Langfuse.**

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![Axum](https://img.shields.io/badge/Axum-0.7-blue?logo=tokio)](https://github.com/tokio-rs/axum)
[![Tokio](https://img.shields.io/badge/Tokio-Async-black?logo=tokio)](https://tokio.rs/)
[![Moka](https://img.shields.io/badge/Moka-In--Memory_Cache-brightgreen)](https://github.com/moka-rs/moka)
[![FastEmbed](https://img.shields.io/badge/FastEmbed-Local_ONNX-purple)](https://github.com/Anush008/fastembed-rs)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Langfuse_OTLP-blueviolet?logo=opentelemetry)](https://langfuse.com)
[![Docker](https://img.shields.io/badge/Docker-Immutable_Multi--Stage-2496ED?logo=docker)](https://www.docker.com/)

</div>

---

## 🚀 Key Features

- **Blazing Fast In-Memory Semantic Caching:** Dual-store architecture pairing `moka` (for TTL & eviction) with a thread-safe cosine similarity vector scanner using local `AllMiniLML6V2` embeddings via ONNX Runtime. Drops repeated/similar prompt latency from **~1,400ms to <7ms (99.5% reduction)**.
- **Full SSE Streaming Support:** Transparently proxies Server-Sent Events with `stream: true`. Replays cached entries as realistic word-by-word streaming responses (`10ms` inter-chunk delay) so client UIs render smoothly.
- **Zero-Copy Stream Interception & Buffering:** On streaming cache misses, bytes are teed to the client in real-time while token deltas are buffered in background memory to synthesize canonical cached entries upon stream completion.
- **OpenTelemetry & Langfuse Integration:** Fully instrumented with OTLP HTTP binary protobuf export to Langfuse. Automatically tracks:
  - `cache.status`: `HIT` vs `MISS`
  - `cache.similarity_score`: Cosine similarity on hits
  - `llm.usage.prompt_tokens` & `llm.usage.completion_tokens`: Extracted from provider JSON & streaming chunks
  - Sub-spans for ONNX vector generation and similarity scanning
- **Immutable, Airgapped Production Docker Image:** Multi-stage Docker build pre-compiles dependencies and bakes the ONNX model weights directly into `/app/fastembed_cache`. Zero runtime downloads required from Hugging Face.
- **Enterprise Security Compliance:** Runs under an unprivileged non-root user (`UID 10001`), handles `SIGTERM`/`SIGINT` graceful shutdown, and flushes pending OTel batches before terminating.

---

## 📐 Architecture Overview

```
Client (OpenAI SDK / curl)
          │
          ▼
   Axum Router (:8080)
          │
    [Stream Mode Check]
     ├── stream: false ──► [Semantic Cache Check] ──┬── HIT  ──► Return Cached JSON (X-Cache: HIT)
     │                                              └── MISS ──► Forward Upstream ──► Cache ──► Return JSON
     │
     └── stream: true  ──► [Semantic Cache Check] ──┬── HIT  ──► Simulated Word-by-Word SSE Replay
                                                    └── MISS ──► Upstream SSE Stream
                                                                  ├─ Tee chunks to client
                                                                  ├─ Buffer content deltas
                                                                  └─ Cache synthetic JSON on [DONE]
```

---

## 🛠️ Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Language** | Rust 2021 Edition | High performance, memory safety, zero-cost abstractions |
| **HTTP Server** | Axum 0.7 + Tokio | Async non-blocking network I/O |
| **HTTP Client** | Reqwest 0.12 (rustls-tls, stream) | Connection-pooled upstream forwarding |
| **Local Embeddings** | FastEmbed 3 (AllMiniLML6V2) | In-process ONNX embeddings (384 dims, ~10ms CPU) |
| **Cache Layer** | Moka 0.12 (future) + RwLock Vector Index | Dual-store TTL, eviction, and cosine similarity scan |
| **Observability** | OpenTelemetry 0.27 + Langfuse | Distributed tracing and token usage accounting |
| **Load Testing** | k6 | High-concurrency performance benchmarking |

---

## ⚡ Quick Start

### 1. Local Development (Cargo)

```bash
# Clone the repository
git clone https://github.com/Hafiz-alt/enterprise-ai-gateway.git
cd enterprise-ai-gateway

# Set environment variables
export OPENAI_API_KEY="sk-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"

# Run in release mode
cargo run --release
```

### 2. Docker (Production Container)

```bash
# Build the self-contained image (pre-downloads ONNX model weights)
docker build -t enterprise-ai-gateway:latest .

# Run the container as non-root user
docker run -d \
  --name ai-gateway \
  -p 8080:8080 \
  -e OPENAI_API_KEY="sk-..." \
  -e LANGFUSE_PUBLIC_KEY="pk-lf-..." \
  -e LANGFUSE_SECRET_KEY="sk-lf-..." \
  enterprise-ai-gateway:latest
```

---

## 🧪 Testing the Endpoints

### Standard Completion (Non-Streaming)
```bash
curl -i http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence."}]
  }'
```
*First request returns `X-Cache: MISS`. Repeating or asking a semantically equivalent query returns `X-Cache: HIT` instantly.*

### Streaming Completion (SSE)
```bash
curl -N -i http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a Python bubble sort algorithm."}]
  }'
```

### Health Check
```bash
curl http://localhost:8080/health
# Output: {"cache_entries":2,"similarity_threshold":0.92,"status":"ok","vector_entries":2}
```

---

## 📊 Performance Benchmarking (k6)

Run the included high-concurrency stress test (50 virtual users):

```bash
# Run with Docker (no local k6 required)
docker run --rm -i \
  -v "${PWD}:/scripts" \
  -e GATEWAY_URL="http://host.docker.internal:8080/v1/chat/completions" \
  grafana/k6 run /scripts/load_test.js
```

### Benchmark Results Under 50 VUs
```text
     ✓ status is 200
     ✓ response body non-empty

     checks.........................: 100.00% ✓ 4800      ✗ 0
     cache_hit_rate.................: 99.90%
   ✓ cache_hit_duration.............: avg=3.4ms   med=2.8ms   p(95)=6.2ms   max=28ms
     cache_miss_duration............: avg=1120ms  med=1090ms  p(95)=1420ms  max=1600ms
     http_req_failed................: 0.00%
```

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(Required)* | OpenAI / upstream API key |
| `UPSTREAM_URL` | `https://api.openai.com/v1/chat/completions` | Target LLM endpoint |
| `GATEWAY_PORT` | `8080` | Port the gateway listens on |
| `SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity threshold for semantic cache hits |
| `CACHE_MAX_CAPACITY` | `10000` | Max entries in Moka and Vector Index |
| `CACHE_TTL_SECS` | `3600` | Cache time-to-live in seconds |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret key |
| `OTEL_EXPORTER_OTLP_ENDPOINT`| `https://cloud.langfuse.com/api/public/otel` | Langfuse OTLP ingest endpoint |
| `FASTEMBED_CACHE_PATH` | `/app/fastembed_cache` | Path to pre-baked ONNX embedding model |

---

## 📄 License

MIT License. Designed and built for high-performance production LLM workloads.
