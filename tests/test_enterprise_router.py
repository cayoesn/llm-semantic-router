import time
import pytest
from app.core.cost_cascade_router import CostAwareCascadeRouter, ModelTier, cascade_router
from app.core.semantic_cache import SemanticCache


def test_cost_cascade_router_simple_query():
    decision = cascade_router.route_query("Qual eh a capital da Franca?")
    assert decision.selected_tier == ModelTier.SLM_NANO
    assert decision.target_model_name == "llama-3.2-3b-instruct"


def test_cost_cascade_router_complex_query():
    decision = cascade_router.route_query("Analise detalhadamente os trade-offs e a arquitetura de sistemas distribuídos sob alto tráfego.")
    assert decision.selected_tier == ModelTier.LLM_FRONTIER
    assert decision.target_model_name == "gpt-4o"


def test_cost_cascade_router_code_detection():
    query = "Como refatorar este método? def calculate_metrics(data): return sum(data)"
    decision = cascade_router.route_query(query)
    assert decision.metrics["has_code"] == 1.0
    assert decision.selected_tier in [ModelTier.SLM_STANDARD, ModelTier.LLM_FRONTIER]


def test_semantic_cache_lru_bounded_eviction():
    # Cache limitado a 3 itens
    cache = SemanticCache(max_size=3)
    
    cache.set("pergunta 1", "resposta 1", "model_1")
    cache.set("pergunta 2", "resposta 2", "model_2")
    cache.set("pergunta 3", "resposta 3", "model_3")
    
    assert len(cache._cache) == 3
    
    # Acessa 'pergunta 1' para torná-la a mais recente no LRU
    entry, sim = cache.search("pergunta 1", threshold=0.8)
    assert entry is not None
    assert entry.prompt == "pergunta 1"
    
    # Adiciona o 4º item -> deve expulsar 'pergunta 2' (que é agora a mais antiga/LRU)
    cache.set("pergunta 4", "resposta 4", "model_4")
    
    assert len(cache._cache) == 3
    assert "pergunta 2" not in cache._cache
    assert "pergunta 1" in cache._cache
    assert "pergunta 3" in cache._cache
    assert "pergunta 4" in cache._cache


def test_semantic_cache_ttl_expiration():
    cache = SemanticCache(max_size=10, default_ttl_seconds=0.05)
    cache.set("prompt rápido", "resposta rápida", "model_quick", ttl_seconds=0.05)
    
    # Busca imediata: deve encontrar
    hit, sim = cache.search("prompt rápido", threshold=0.8)
    assert hit is not None
    
    # Aguarda TTL expirar
    time.sleep(0.06)
    
    # Segunda busca: deve ignorar/purgar expirado
    miss, sim = cache.search("prompt rápido", threshold=0.8)
    assert miss is None
