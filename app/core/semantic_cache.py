import time
import logging
from collections import OrderedDict
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from app.core.embeddings import embedder
from app.config import settings

logger = logging.getLogger("llm_semantic_router.cache")


class CacheEntry:
    def __init__(
        self,
        prompt: str,
        vector: List[float],
        response: str,
        model_used: str,
        cost_saved: float = 0.001,
        ttl_seconds: float = 86400.0,
    ):
        self.prompt = prompt
        self.vector = vector
        self.response = response
        self.model_used = model_used
        self.cost_saved = cost_saved
        self.created_at = time.time()
        self.timestamp = self.created_at
        self.last_accessed_at = self.created_at
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class SemanticCache:
    """Enterprise Bounded LRU & Matrix-Accelerated Semantic Vector Cache.
    
    Características Staff/Principal:
    1. Bounded LRU com OrderedDict: Limite estrito de memória (max_size) prevenindo OOM.
    2. Matrix Acceleration: Projeção vetorial BLAS simultânea (O(N) em C SIMD em vez de loops Python).
    3. Suporte a TTL por entrada para invalidação e purga assíncrona/on-access.
    4. Métricas de Hit Ratio, Latência economizada e Projeção Financeira USD.
    """

    def __init__(self, max_size: int = 10000, default_ttl_seconds: float = 86400.0):
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._vectors_cache: List[np.ndarray] = []
        self._keys_cache: List[str] = []
        
        self._total_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_saved_usd: float = 0.0

    def _sync_vector_matrix(self):
        """Reconstrói buffers de busca linear indexada."""
        self._keys_cache = list(self._cache.keys())
        self._vectors_cache = [
            np.array(self._cache[k].vector, dtype=np.float32) for k in self._keys_cache
        ]

    def search(self, prompt: str, threshold: float = 0.88) -> Tuple[Optional[CacheEntry], float]:
        """Busca vetorizada no cache com suporte a BLAS e verificação de expiração TTL."""
        self._total_requests += 1
        query_vector = np.array(embedder.encode(prompt), dtype=np.float32)
        norm_q = np.linalg.norm(query_vector)
        if norm_q > 0:
            query_vector = query_vector / norm_q

        if not self._cache:
            self._cache_misses += 1
            return None, 0.0

        # Purga sob demanda se encontrar expirados na busca
        now = time.time()
        valid_keys = []
        valid_vectors = []
        for key, entry in list(self._cache.items()):
            if entry.is_expired:
                del self._cache[key]
            else:
                valid_keys.append(key)
                valid_vectors.append(entry.vector)

        if not valid_keys:
            self._cache_misses += 1
            return None, 0.0

        # Matrix dot product acelerado: V (N, D) . q (D,) -> similarities (N,)
        matrix = np.array(valid_vectors, dtype=np.float32)
        # Assegura normalização das linhas
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        norm_matrix = matrix / row_norms

        similarities = np.dot(norm_matrix, query_vector)
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_key = valid_keys[best_idx]
        best_entry = self._cache[best_key]

        if best_similarity >= threshold:
            best_entry.hit_count += 1
            best_entry.last_accessed_at = now
            self._cache.move_to_end(best_key)  # Marca como recentemente usado (LRU)
            self._cache_hits += 1
            self._total_saved_usd += best_entry.cost_saved
            logger.info(f"Cache HIT! Similarity: {best_similarity:.4f} >= {threshold}")
            return best_entry, best_similarity

        self._cache_misses += 1
        logger.info(f"Cache MISS. Best similarity: {best_similarity:.4f} < {threshold}")
        return None, best_similarity

    def set(
        self,
        prompt: str,
        response: str,
        model_used: str,
        cost_saved: float = 0.001,
        ttl_seconds: Optional[float] = None,
    ) -> CacheEntry:
        """Adiciona uma entrada no cache com política de expulsão LRU quando atingir max_size."""
        # Se a chave já existir, atualiza
        if prompt in self._cache:
            self._cache.move_to_end(prompt)
            entry = self._cache[prompt]
            entry.response = response
            entry.model_used = model_used
            entry.cost_saved = cost_saved
            entry.created_at = time.time()
            return entry

        # Se atingir max_size, expulsa a entrada menos recentemente usada (FIFO/LRU no início)
        if len(self._cache) >= self.max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug(f"Cache eviction: {evicted_key} purged by LRU policy.")

        vector = embedder.encode(prompt)
        entry = CacheEntry(
            prompt=prompt,
            vector=vector,
            response=response,
            model_used=model_used,
            cost_saved=cost_saved,
            ttl_seconds=ttl_seconds or self.default_ttl_seconds,
        )
        self._cache[prompt] = entry
        return entry

    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas analíticas do cache semântico."""
        hit_ratio = (self._cache_hits / self._total_requests * 100.0) if self._total_requests > 0 else 0.0
        return {
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_ratio_percentage": round(hit_ratio, 2),
            "total_saved_usd": round(self._total_saved_usd, 6),
            "cached_prompts_count": len(self._cache),
            "max_size": self.max_size,
        }

    def clear(self):
        """Limpa o cache semântico."""
        self._cache.clear()
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_saved_usd = 0.0


semantic_cache = SemanticCache()
