import time
import logging
from typing import Optional, Dict, Any, List, Tuple
from app.core.embeddings import embedder
from app.config import settings

logger = logging.getLogger("llm_semantic_router.cache")


class CacheEntry:
    def __init__(self, prompt: str, vector: List[float], response: str, model_used: str, cost_saved: float = 0.001):
        self.prompt = prompt
        self.vector = vector
        self.response = response
        self.model_used = model_used
        self.cost_saved = cost_saved
        self.timestamp = time.time()
        self.hit_count = 0


class SemanticCache:
    """
    Mecanismo de Cache Semântico vetorial em memória com busca por similaridade de cosseno.
    """
    def __init__(self):
        self._cache: List[CacheEntry] = []
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_saved_usd: float = 0.0

    def search(self, prompt: str, threshold: float = 0.88) -> Tuple[Optional[CacheEntry], float]:
        """
        Busca uma entrada no cache cuja similaridade vetorial seja maior ou igual ao threshold.
        Retorna (CacheEntry ou None, similarity_score).
        """
        self._total_requests += 1
        query_vector = embedder.encode(prompt)
        
        best_entry: Optional[CacheEntry] = None
        best_similarity: float = 0.0

        for entry in self._cache:
            sim = embedder.cosine_similarity(query_vector, entry.vector)
            if sim > best_similarity:
                best_similarity = sim
                best_entry = entry

        if best_entry and best_similarity >= threshold:
            best_entry.hit_count += 1
            self._cache_hits += 1
            self._total_saved_usd += best_entry.cost_saved
            logger.info(f"Cache HIT! Similarity: {best_similarity:.4f} >= {threshold}")
            return best_entry, best_similarity

        self._cache_misses += 1
        logger.info(f"Cache MISS. Best similarity: {best_similarity:.4f} < {threshold}")
        return None, best_similarity

    def set(self, prompt: str, response: str, model_used: str, cost_saved: float = 0.001) -> CacheEntry:
        """
        Adiciona uma nova resposta e seu vetor ao cache semântico.
        """
        vector = embedder.encode(prompt)
        entry = CacheEntry(
            prompt=prompt,
            vector=vector,
            response=response,
            model_used=model_used,
            cost_saved=cost_saved
        )
        self._cache.append(entry)
        return entry

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas detalhadas do uso do cache semântico.
        """
        hit_ratio = (self._cache_hits / self._total_requests * 100.0) if self._total_requests > 0 else 0.0
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio_percentage": round(hit_ratio, 2),
            "total_saved_usd": round(self._total_saved_usd, 6),
            "cached_prompts_count": len(self._cache)
        }

    def clear(self):
        """
        Limpa todas as entradas do cache semântico.
        """
        self._cache.clear()
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_saved_usd = 0.0


semantic_cache = SemanticCache()
