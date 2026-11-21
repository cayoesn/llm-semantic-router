import time
import pytest
from app.core.cost_cascade_router import CostAwareCascadeRouter, ModelTier, cascade_router
from app.core.semantic_cache import SemanticCache


def test_cost_cascade_router_simple_query():
    decision = cascade_router.route_query("What is the capital of France?")
    assert decision.selected_tier == ModelTier.SLM_NANO
    assert decision.target_model_name == "llama-3.2-3b-instruct"


def test_cost_cascade_router_complex_query():
    decision = cascade_router.route_query(
        "Analyze in detail the trade-offs and system architecture of distributed services under high traffic."
    )
    assert decision.selected_tier == ModelTier.LLM_FRONTIER
    assert decision.target_model_name == "gpt-4o"


def test_cost_cascade_router_code_detection():
    query = "How can I refactor this method? def calculate_metrics(data): return sum(data)"
    decision = cascade_router.route_query(query)
    assert decision.metrics["has_code"] == 1.0
    assert decision.selected_tier in [ModelTier.SLM_STANDARD, ModelTier.LLM_FRONTIER]


def test_semantic_cache_lru_bounded_eviction():
    # Cache bounded to 3 items
    cache = SemanticCache(max_size=3)

    cache.set("question 1", "answer 1", "model_1")
    cache.set("question 2", "answer 2", "model_2")
    cache.set("question 3", "answer 3", "model_3")

    assert len(cache._cache) == 3

    # Access 'question 1' so it becomes the most recently used entry (LRU)
    entry, sim = cache.search("question 1", threshold=0.8)
    assert entry is not None
    assert entry.prompt == "question 1"

    # Insert 4th item -> must evict 'question 2' (now the least recently used)
    cache.set("question 4", "answer 4", "model_4")

    assert len(cache._cache) == 3
    assert "question 2" not in cache._cache
    assert "question 1" in cache._cache
    assert "question 3" in cache._cache
    assert "question 4" in cache._cache


def test_semantic_cache_ttl_expiration():
    cache = SemanticCache(max_size=10, default_ttl_seconds=0.05)
    cache.set("fast prompt", "fast response", "model_quick", ttl_seconds=0.05)

    # Immediate lookup: must hit
    hit, sim = cache.search("fast prompt", threshold=0.8)
    assert hit is not None

    # Wait for TTL expiration
    time.sleep(0.06)

    # Second lookup: must ignore and purge expired entry
    miss, sim = cache.search("fast prompt", threshold=0.8)
    assert miss is None
