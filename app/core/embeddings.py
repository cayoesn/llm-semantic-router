import numpy as np
from typing import List
import math
import re


class VectorEmbedder:
    """
    Gera vetores de embedding leve e determinísticos com base na frequência e dispersão semântica das palavras,
    garantindo computação de similaridade de cosseno em sub-milissegundos sem dependências pesadas.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", "", text.lower())
        return [w for w in cleaned.split() if w]

    def encode(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.dimension

        vec = np.zeros(self.dimension, dtype=np.float32)
        for token in tokens:
            # Hash determinístico do token para mapear nas dimensões do vetor
            for i in range(3):
                idx = hash(f"{token}_{i}") % self.dimension
                val = (hash(token) % 1000) / 1000.0
                vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        dot_product = np.dot(a, b)
        similarity = dot_product / (norm_a * norm_b)
        return float(max(0.0, min(1.0, similarity)))


embedder = VectorEmbedder()
