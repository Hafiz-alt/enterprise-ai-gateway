# ==============================================================================
# Enterprise AI Gateway — Production Multi-Stage Dockerfile
# ==============================================================================
#
# Features:
#   1. Layer-cached compilation: Dependencies are compiled separately from source.
#   2. Completely immutable: FastEmbed AllMiniLML6V2 ONNX weights pre-baked into
#      image at build-time. Container starts instantly with ZERO internet access
#      required for model initialization.
#   3. Hardened security: Runs as non-root unprivileged user (UID 10001).
#   4. Minimal attack surface: Clean debian:bookworm-slim runner with only
#      required runtime shared libraries and CA certificates.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Model Pre-Download
# ------------------------------------------------------------------------------
FROM rust:1.82-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 1. Cache dependencies: Copy only manifests first
COPY Cargo.toml Cargo.lock* ./

# 2. Stub out source tree and compile dependencies to cache them in Docker layer
RUN mkdir -p src/bin && \
    echo "fn main() {}" > src/main.rs && \
    echo "fn main() {}" > src/bin/pre_download.rs && \
    cargo build --release && \
    rm -rf src

# 3. Copy real source code
COPY src ./src

# Set target directory for build-time model download
ENV FASTEMBED_CACHE_PATH=/app/fastembed_cache \
    FASTEMBED_CACHE_DIR=/app/fastembed_cache

# 4. Build actual release binaries
RUN touch src/main.rs src/bin/pre_download.rs && \
    cargo build --release --bin ai-gateway --bin pre_download

# 5. Pre-download the FastEmbed AllMiniLML6V2 ONNX model & tokenizer files
RUN ./target/release/pre_download

# ------------------------------------------------------------------------------
# Stage 2: Production Minimal Runner
# ------------------------------------------------------------------------------
FROM debian:bookworm-slim AS runner

# Install TLS certificates (needed for outbound HTTPS to OpenAI and Langfuse)
# and clean up apt cache immediately to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create unprivileged system user and group for enterprise compliance
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -d /app -s /usr/sbin/nologin -M appuser

WORKDIR /app

# Copy the pre-downloaded ONNX model cache from builder stage
COPY --from=builder --chown=appuser:appgroup /app/fastembed_cache /app/fastembed_cache

# Copy the compiled production binary
COPY --from=builder --chown=appuser:appgroup /build/target/release/ai-gateway /usr/local/bin/ai-gateway

# Set runtime permissions
RUN chown -R appuser:appgroup /app && \
    chmod +x /usr/local/bin/ai-gateway

# Runtime environment variables
ENV FASTEMBED_CACHE_PATH=/app/fastembed_cache \
    FASTEMBED_CACHE_DIR=/app/fastembed_cache \
    GATEWAY_PORT=8080 \
    RUST_LOG=ai_gateway=info,tower_http=debug,warn

# Drop privileges to non-root user
USER appuser:appgroup

# Expose HTTP port
EXPOSE 8080

# Run the AI Gateway
ENTRYPOINT ["/usr/local/bin/ai-gateway"]
