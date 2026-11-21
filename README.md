# 🔀 LLM Semantic Router (Enterprise Edition)

Intelligent Semantic Routing Engine, **Cascade Model Selector**, and **Bounded LRU Vector Semantic Cache** with BLAS/SIMD matrix acceleration.

---

## 🌟 Architecture & Enterprise Capabilities

- **Bounded LRU Semantic Cache**:
  - Strictly memory-bounded storage (`max_size`) backed by `collections.OrderedDict`, guaranteeing $\mathcal{O}(1)$ eviction and preventing memory leaks / OOM failures.
  - Per-entry TTL (Time-To-Live) expiration with asynchronous and on-demand purging.
- **Matrix-Accelerated Vector Search**:
  - Batch similarity search projecting the query vector onto the full embedding matrix $\mathbf{V} \in \mathbb{R}^{N \times D}$:
    $$\mathbf{s} = \mathbf{V}_{\text{norm}} \cdot \mathbf{q}_{\text{norm}}^T$$
  - Executed at native C speed via NumPy BLAS/SIMD with zero Python loop overhead.
- **Cost-Aware Cascade Model Selector**:
  - Multi-dimensional complexity analysis (Type-Token Ratio, lexical markers, and code syntax detection) dynamically routing across:
    - **SLM Nano** (`llama-3.2-3b-instruct`): Fast, conversational, low-cost queries (\$0.00005).
    - **SLM Standard** (`llama-3.3-70b-instruct`): Intermediate technical queries (\$0.0005).
    - **LLM Frontier** (`gpt-4o`): Complex syntheses, algorithms, and architectural trade-offs (\$0.005).
- **OTel, Langfuse & Prometheus Observability**:
  - End-to-end tracking of cache hit vs. miss latency and continuous projection of accumulated financial savings (USD).

---

## 📊 Algorithmic Complexity

| Operation | Legacy Implementation | Optimized Implementation (Staff) | Data Structure |
| :--- | :--- | :--- | :--- |
| **Cache Eviction** | None (unbounded growth) | $\mathcal{O}(1)$ Bounded LRU | `OrderedDict.popitem(last=False)` |
| **Similarity Search** | $\mathcal{O}(N \times D)$ scalar Python loops | $\mathcal{O}(N \times D)$ NumPy Matrix Dot | 2D Contiguous Buffer |
| **Complexity Classification** | Simple token count | Lexical + Syntactic + TTR Analysis | Multi-Signal Classifier |

---

## 🚀 Running with Docker

```bash
docker compose up -d --build
```

---

## 🧪 Automated Tests (12 Tests / 100% Hermetic / 94% Coverage)

Hermetic execution of the test suite:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace portfolio-spec-engine:test pytest tests -v --cov=app/core
```
