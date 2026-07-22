import logging
from typing import Any, Dict, Optional
from app.config import settings

logger = logging.getLogger("llm_semantic_router.observability")

try:
    from langfuse import Langfuse
    langfuse_client = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )
except Exception as e:
    logger.warning(f"Could not initialize Langfuse in semantic router: {e}")
    langfuse_client = None


def _create_trace(
    name: str,
    session_id: Optional[str] = None,
    tags: Optional[list] = None,
    input_data: Optional[Any] = None,
    metadata: Optional[dict] = None
) -> Any:
    if not langfuse_client:
        return None
    try:
        if hasattr(langfuse_client, "trace"):
            return langfuse_client.trace(
                name=name,
                session_id=session_id,
                tags=tags,
                input=input_data,
                metadata=metadata
            )
        elif hasattr(langfuse_client, "start_observation"):
            return langfuse_client.start_observation(
                name=name,
                type="TRACE",
                session_id=session_id,
                tags=tags,
                input=input_data,
                metadata=metadata
            )
    except Exception as e:
        logger.warning(f"Error creating Langfuse trace: {e}")
    return None


class RouterObservability:
    @staticmethod
    def trace_routing_request(
        prompt: str,
        similarity_threshold: float,
        session_id: Optional[str] = None
    ) -> Any:
        return _create_trace(
            name="semantic_routing_decision",
            session_id=session_id or "router-anonymous-session",
            tags=["llm-semantic-router", "cache-and-routing"],
            input_data={"prompt": prompt, "similarity_threshold": similarity_threshold},
            metadata={"threshold": similarity_threshold}
        )

    @staticmethod
    def record_span(
        trace: Any,
        span_name: str,
        input_data: Dict[str, Any],
        output_data: Optional[Dict[str, Any]] = None
    ) -> Any:
        if not trace:
            return None
        try:
            if hasattr(trace, "span"):
                span = trace.span(name=span_name, input=input_data)
                if output_data and hasattr(span, "end"):
                    span.end(output=output_data)
                return span
        except Exception as e:
            logger.warning(f"Error recording span: {e}")
        return None

    @staticmethod
    def record_metrics(
        trace: Any,
        cache_hit: bool,
        target_model: str,
        similarity: float,
        latency_ms: float,
        saved_usd: float
    ):
        if not trace:
            return
        try:
            if hasattr(trace, "update"):
                trace.update(
                    output={
                        "cache_hit": cache_hit,
                        "target_model": target_model,
                        "similarity_score": similarity,
                        "latency_ms": latency_ms,
                        "saved_usd": saved_usd
                    },
                    metadata={
                        "cache_status": "HIT" if cache_hit else "MISS",
                        "model": target_model,
                        "saved_usd": saved_usd
                    }
                )
            if hasattr(trace, "score"):
                trace.score(name="similarity_score", value=similarity)
                trace.score(name="saved_usd", value=saved_usd)
                trace.score(name="latency_ms", value=latency_ms)
        except Exception as e:
            logger.warning(f"Error recording metrics on trace: {e}")

    @staticmethod
    def flush():
        if langfuse_client and hasattr(langfuse_client, "flush"):
            try:
                langfuse_client.flush()
            except Exception:
                pass
