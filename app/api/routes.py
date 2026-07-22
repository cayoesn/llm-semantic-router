import time
from fastapi import APIRouter, HTTPException, status
from app.api.schemas import RouteRequest, RouteResponse, CacheStatsResponse, MessageResponse
from app.core.semantic_cache import semantic_cache
from app.core.semantic_router import semantic_router
from app.core.observability import RouterObservability
from app.config import settings

router = APIRouter()


@router.post("/api/v1/route", response_model=RouteResponse, status_code=status.HTTP_200_OK)
async def route_and_execute(req: RouteRequest):
    """
    Roteia o prompt do usuário avaliando o cache semântico (Cache HIT < 5ms) e seleciona
    o modelo ideal de destino (Tier-1 Flash vs Tier-2 Pro) caso haja Cache MISS.
    """
    start_time = time.time()
    threshold = req.similarity_threshold if req.similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    # Rastreamento de Observabilidade no Langfuse
    trace = RouterObservability.trace_routing_request(
        prompt=req.prompt,
        similarity_threshold=threshold,
        session_id=req.session_id
    )

    # Step 1: Checa Cache Semântico
    cached_entry, similarity = None, 0.0
    if not req.force_refresh:
        span_cache = RouterObservability.record_span(trace, "cache_lookup", {"threshold": threshold})
        cached_entry, similarity = semantic_cache.search(req.prompt, threshold=threshold)
        if span_cache:
            RouterObservability.record_span(
                trace,
                "cache_lookup",
                {"threshold": threshold},
                {"hit": cached_entry is not None, "similarity": similarity}
            )

    if cached_entry:
        latency_ms = (time.time() - start_time) * 1000.0
        saved_usd = cached_entry.cost_saved

        RouterObservability.record_metrics(
            trace=trace,
            cache_hit=True,
            target_model=cached_entry.model_used,
            similarity=similarity,
            latency_ms=latency_ms,
            saved_usd=saved_usd
        )
        RouterObservability.flush()

        return RouteResponse(
            prompt=req.prompt,
            response=f"[CACHE HIT ⚡] {cached_entry.response}",
            target_model=cached_entry.model_used,
            tier_level="Semantic-Cache",
            cache_hit=True,
            similarity_score=round(similarity, 4),
            latency_ms=round(latency_ms, 2),
            saved_usd=saved_usd,
            metadata={"source": "semantic_cache", "cached_at": cached_entry.timestamp}
        )

    # Step 2: Cache MISS - Executa Roteador Semântico
    span_route = RouterObservability.record_span(trace, "semantic_routing_classification", {"prompt": req.prompt})
    target_model, tier_level, route_metadata = semantic_router.classify_and_route(req.prompt)
    if span_route:
        RouterObservability.record_span(
            trace,
            "semantic_routing_classification",
            {"prompt": req.prompt},
            {"target_model": target_model, "tier": tier_level}
        )

    # Simula geração do modelo de linguagem (Gera resposta coerente e armazena no cache)
    simulated_response = f"Resposta processada via modelo {target_model} [{tier_level}] para a consulta: '{req.prompt}'"
    cost_saved = settings.COST_PER_1K_ADVANCED if "pro" in target_model else settings.COST_PER_1K_FAST
    
    # Atualiza o cache com o novo vetor e resposta
    semantic_cache.set(
        prompt=req.prompt,
        response=simulated_response,
        model_used=target_model,
        cost_saved=cost_saved
    )

    latency_ms = (time.time() - start_time) * 1000.0

    RouterObservability.record_metrics(
        trace=trace,
        cache_hit=False,
        target_model=target_model,
        similarity=similarity,
        latency_ms=latency_ms,
        saved_usd=0.0
    )
    RouterObservability.flush()

    return RouteResponse(
        prompt=req.prompt,
        response=simulated_response,
        target_model=target_model,
        tier_level=tier_level,
        cache_hit=False,
        similarity_score=round(similarity, 4),
        latency_ms=round(latency_ms, 2),
        saved_usd=0.0,
        metadata=route_metadata
    )


@router.get("/api/v1/cache/stats", response_model=CacheStatsResponse, status_code=status.HTTP_200_OK)
async def get_cache_statistics():
    """
    Retorna métricas de hit ratio, economia estimada em USD e total de requisições.
    """
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete("/api/v1/cache", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def clear_semantic_cache():
    """
    Expurga todas as entradas e zera as estatísticas do cache semântico.
    """
    semantic_cache.clear()
    return MessageResponse(message="Semantic cache cleared successfully", status="success")


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Endpoint de verificação de saúde do serviço.
    """
    return {
        "status": "HEALTHY",
        "service": settings.APP_NAME,
        "cached_prompts": len(semantic_cache._cache)
    }
