import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.embeddings import embedder, VectorEmbedder
from app.core.semantic_cache import semantic_cache, SemanticCache
from app.core.semantic_router import semantic_router
from app.core.observability import RouterObservability, _create_trace
import app.core.observability as obs_mod

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_cache():
    semantic_cache.clear()
    yield
    semantic_cache.clear()


def test_vector_embedder():
    vec1 = embedder.encode("Como configurar FastAPI?")
    vec2 = embedder.encode("Como criar uma API com FastAPI?")
    vec_empty = embedder.encode("")

    assert len(vec1) == 384
    assert len(vec2) == 384
    assert len(vec_empty) == 384

    sim = VectorEmbedder.cosine_similarity(vec1, vec2)
    assert 0.0 <= sim <= 1.0

    # Teste de vetor zero
    sim_zero = VectorEmbedder.cosine_similarity([0.0] * 384, [1.0] * 384)
    assert sim_zero == 0.0


def test_semantic_cache_operations():
    cache = SemanticCache()
    assert cache.get_stats()["total_requests"] == 0

    # Adiciona entrada
    cache.set(
        prompt="Qual é o preço do arroz?",
        response="O arroz custa R$ 5,00/kg",
        model_used="gemini-3.5-flash",
        cost_saved=0.0005
    )

    # Busca com similaridade alta (HIT)
    entry_hit, sim_hit = cache.search("Qual é o preço do arroz?", threshold=0.80)
    assert entry_hit is not None
    assert "arroz" in entry_hit.response
    assert sim_hit >= 0.80

    # Busca com prompt diferente (MISS)
    entry_miss, sim_miss = cache.search("Como funciona a teoria da relatividade?", threshold=0.95)
    assert entry_miss is None

    stats = cache.get_stats()
    assert stats["total_requests"] == 2
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["hit_ratio_percentage"] == 50.0

    cache.clear()
    assert cache.get_stats()["cached_prompts_count"] == 0


def test_semantic_router_classification():
    # Prompt simples -> Tier 1
    model_simple, tier_simple, meta_simple = semantic_router.classify_and_route("Olá, bom dia!")
    assert "flash" in model_simple
    assert "Tier-1" in tier_simple

    # Prompt complexo com palavras-chave -> Tier 2
    model_complex, tier_complex, meta_complex = semantic_router.classify_and_route("Como refatorar e otimizar um algoritmo em Python para arquitetura de microsserviços?")
    assert "pro" in model_complex
    assert "Tier-2" in tier_complex


def test_api_route_and_cache_flow():
    payload = {
        "prompt": "Qual o horário de funcionamento da loja?",
        "session_id": "test-session-1",
        "force_refresh": False,
        "similarity_threshold": 0.85
    }

    # 1a Chamada -> Cache MISS (Insere no Cache)
    response1 = client.post("/api/v1/route", json=payload)
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["cache_hit"] is False
    assert "gemini-3.5-flash" in data1["target_model"]

    # 2a Chamada -> Cache HIT (< 5ms)
    response2 = client.post("/api/v1/route", json=payload)
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["cache_hit"] is True
    assert "[CACHE HIT ⚡]" in data2["response"]
    assert data2["similarity_score"] >= 0.85

    # 3a Chamada com force_refresh -> Cache MISS forçado
    payload_refresh = {**payload, "force_refresh": True}
    response3 = client.post("/api/v1/route", json=payload_refresh)
    assert response3.status_code == 200
    assert response3.json()["cache_hit"] is False


def test_api_cache_stats_and_purge():
    # Popula cache
    client.post("/api/v1/route", json={"prompt": "Pergunta A"})
    client.post("/api/v1/route", json={"prompt": "Pergunta A"})  # Hit

    stats_res = client.get("/api/v1/cache/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert stats["total_requests"] == 2
    assert stats["cache_hits"] == 1

    # Limpa cache
    purge_res = client.delete("/api/v1/cache")
    assert purge_res.status_code == 200
    assert purge_res.json()["status"] == "success"

    # Verifica cache zerado
    stats_after = client.get("/api/v1/cache/stats").json()
    assert stats_after["cached_prompts_count"] == 0


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "LLM Semantic Router" in data["service"]


def test_observability_coverage(monkeypatch):
    class MockSpan:
        def end(self, output=None):
            pass

    class MockTrace:
        def span(self, name, input=None):
            return MockSpan()
        def update(self, output=None, metadata=None):
            pass
        def score(self, name, value):
            pass

    class MockLangfuse:
        def trace(self, **kwargs):
            return MockTrace()
        def start_observation(self, **kwargs):
            return MockTrace()
        def flush(self):
            pass

    monkeypatch.setattr(obs_mod, "langfuse_client", MockLangfuse())

    trace = RouterObservability.trace_routing_request("Test Prompt", 0.88)
    assert trace is not None

    span = RouterObservability.record_span(trace, "test_span", {"in": "1"}, {"out": "ok"})
    assert span is not None

    RouterObservability.record_metrics(trace, True, "gemini-3.5-flash", 0.95, 2.5, 0.001)
    RouterObservability.flush()

    # Testes quando langfuse_client é Nulo
    monkeypatch.setattr(obs_mod, "langfuse_client", None)
    assert _create_trace("test") is None
    assert RouterObservability.record_span(None, "span", {}) is None
    RouterObservability.record_metrics(None, False, "model", 0.0, 10.0, 0.0)
