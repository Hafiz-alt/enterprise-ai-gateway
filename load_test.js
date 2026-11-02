// =============================================================================
// load_test.js — High-Concurrency k6 Load Testing Script for AI Gateway
// =============================================================================
//
// Demonstrates ultra-low-latency semantic caching under high load:
//   - Initial requests hit OpenAI (Cache MISS: ~500ms - 2000ms latency)
//   - Subsequent requests hit in-memory Semantic Cache (Cache HIT: ~1ms - 15ms latency)
//   - Tracks separate k6 Trend metrics: `cache_hit_duration` vs `cache_miss_duration`
//
// -----------------------------------------------------------------------------
// HOW TO RUN:
// -----------------------------------------------------------------------------
//
// 1. With Native k6 installed:
//    k6 run load_test.js
//
// 2. With Docker (Zero installation required on Windows / Mac / Linux):
//
//    • Windows PowerShell:
//      docker run --rm -i -v "${PWD}:/scripts" -e GATEWAY_URL="http://host.docker.internal:8080/v1/chat/completions" grafana/k6 run /scripts/load_test.js
//
//    • Linux (using host networking):
//      docker run --rm -i --network=host -v "$PWD:/scripts" grafana/k6 run /scripts/load_test.js
//
// =============================================================================

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

// ── 1. Custom Performance Metrics ───────────────────────────────────────────
// These trends isolate Cache Hits from Cache Misses so you can report the
// exact latency reduction (e.g. 1200ms -> 4ms = 99.6% latency drop).
export const cacheHitDuration = new Trend('cache_hit_duration', true);
export const cacheMissDuration = new Trend('cache_miss_duration', true);
export const cacheHitRate = new Rate('cache_hit_rate');
export const cacheHitCount = new Counter('cache_hits_total');
export const cacheMissCount = new Counter('cache_misses_total');

// ── 2. Load Profile (Ramping VUs) ───────────────────────────────────────────
export const options = {
  stages: [
    { duration: '10s', target: 50 }, // Ramp up to 50 concurrent virtual users
    { duration: '30s', target: 50 }, // Hold high load at 50 VUs for 30s
    { duration: '10s', target: 0 },  // Graceful ramp-down to 0 VUs
  ],
  thresholds: {
    // 95% of cached requests should complete in under 50ms
    'cache_hit_duration': ['p(95)<50', 'p(99)<100'],
    // Overall HTTP failure rate must stay under 1%
    'http_req_failed': ['rate<0.01'],
    // Cache hit rate should approach >85% due to constrained prompt pool
    'cache_hit_rate': ['rate>0.80'],
  },
};

// ── 3. Prompt Pool for Semantic Cache Testing ───────────────────────────────
// A pool of 5 diverse queries. The first 5 requests encounter a cache MISS;
// all subsequent requests trigger fast semantic vector matches (HIT).
const PROMPT_POOL = [
  'Write a Python function to perform an in-place bubble sort algorithm.',
  'Explain quantum computing and quantum superposition in simple terms for beginners.',
  'How does recursive DNS resolution work across root, TLD, and authoritative servers?',
  'What is a mutex and how does it prevent data races in multithreaded programming?',
  'Explain the fundamental architectural differences between TCP and UDP protocols.',
];

// Target gateway endpoint (defaults to localhost:8080, overridable via GATEWAY_URL env var)
const GATEWAY_URL = __ENV.GATEWAY_URL || 'http://localhost:8080/v1/chat/completions';

export default function () {
  // Randomly pick a prompt from our pool
  const prompt = PROMPT_POOL[Math.floor(Math.random() * PROMPT_POOL.length)];

  // OpenAI-compatible Chat Completion JSON body
  const payload = JSON.stringify({
    model: 'gpt-4o',
    messages: [
      {
        role: 'system',
        content: 'You are a helpful software engineering assistant.',
      },
      {
        role: 'user',
        content: prompt,
      },
    ],
    temperature: 0.7,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer sk-1234567890abcdef', // Proxy verifies or passes bearer token
    },
    timeout: '30s',
  };

  // Dispatch POST request to the Gateway
  const res = http.post(GATEWAY_URL, payload, params);

  // Validate HTTP 200 response status
  const isStatus200 = check(res, {
    'status is 200': (r) => r.status === 200,
    'response body non-empty': (r) => r.body && r.body.length > 0,
  });

  // Extract X-Cache header (k6 normalizes headers to Title-Case or lower-case)
  const cacheHeader = res.headers['X-Cache'] || res.headers['x-cache'];

  if (cacheHeader === 'HIT') {
    cacheHitDuration.add(res.timings.duration);
    cacheHitRate.add(1);
    cacheHitCount.add(1);
  } else if (cacheHeader === 'MISS') {
    cacheMissDuration.add(res.timings.duration);
    cacheHitRate.add(0);
    cacheMissCount.add(1);
  }

  // Small pacing sleep between iterations per virtual user
  sleep(0.05);
}
