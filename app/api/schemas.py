from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class RouteRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Prompt submitted by the user",
        examples=["How do I build a FastAPI service with Docker?"],
    )
    session_id: Optional[str] = Field(None, description="Optional user session identifier")
    force_refresh: bool = Field(
        False,
        description="If True, bypasses the semantic cache and forces a model invocation",
    )
    similarity_threshold: Optional[float] = Field(
        0.88,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold required to qualify as a cache HIT",
    )


class RouteResponse(BaseModel):
    prompt: str
    response: str
    target_model: str
    tier_level: str
    cache_hit: bool
    similarity_score: float
    latency_ms: float
    saved_usd: float
    metadata: Dict[str, Any]


class CacheStatsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_ratio_percentage: float
    total_saved_usd: float
    cached_prompts_count: int


class MessageResponse(BaseModel):
    message: str
    status: str
