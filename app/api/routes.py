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
    Routes the user prompt by checking the semantic cache (Cache HIT < 5ms) and selecting
    the optimal target model (Tier-1 Flash vs. Tier-2 Pro) on a Cache MISS.
    """
    start_time = time.time()
    threshold = req.similarity_threshold if req.similarity_threshold is not None else settings.SIMILARITY_THRESHOLD

    # Langfuse observability tracing
    trace = RouterObservability.trace_routing_request(
        prompt=req.prompt,
        similarity_threshold=threshold,
        session_id=req.session_id
    )

    # Step 1: Check semantic cache
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

    # Step 2: Cache MISS - Invoke semantic router
    span_route = RouterObservability.record_span(trace, "semantic_routing_classification", {"prompt": req.prompt})
    target_model, tier_level, route_metadata = semantic_router.classify_and_route(req.prompt)
    if span_route:
        RouterObservability.record_span(
            trace,
            "semantic_routing_classification",
            {"prompt": req.prompt},
            {"target_model": target_model, "tier": tier_level}
        )

    # Simulate LLM generation (produce a coherent response and store it in the cache)
    simulated_response = f"Response processed via model {target_model} [{tier_level}] for query: '{req.prompt}'"
    cost_saved = settings.COST_PER_1K_ADVANCED if "pro" in target_model else settings.COST_PER_1K_FAST
    
    # Update cache with the new embedding vector and response
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
    Returns hit ratio metrics, estimated USD savings, and total request count.
    """
    stats = semantic_cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete("/api/v1/cache", response_model=MessageResponse, status_code=status.HTTP_200_OK)
async def clear_semantic_cache():
    """
    Purges all entries and resets semantic cache statistics.
    """
    semantic_cache.clear()
    return MessageResponse(message="Semantic cache cleared successfully", status="success")


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Service health check endpoint.
    """
    return {
        "status": "HEALTHY",
        "service": settings.APP_NAME,
        "cached_prompts": len(semantic_cache._cache)
    }
