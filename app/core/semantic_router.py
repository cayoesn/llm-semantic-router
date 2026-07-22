import re
import logging
from typing import Dict, Any, Tuple
from app.config import settings

logger = logging.getLogger("llm_semantic_router.router")


class SemanticRouter:
    """
    Roteador de Intenções Semânticas que analisa a complexidade do prompt para selecionar
    a classe adequada de modelo LLM (Fast Flash vs Advanced Pro).
    """
    
    # Palavras-chave indicativas de alta complexidade técnica ou raciocínio profundo
    ADVANCED_KEYWORDS = [
        "código", "code", "refatorar", "refactor", "arquitetura", "architecture",
        "algoritmo", "algorithm", "otimizar", "optimize", "matemática", "math",
        "análise", "analysis", "debug", "sql", "python", "docker", "fastapi",
        "microservices", "provar", "proof", "raciocínio", "reasoning"
    ]

    def classify_and_route(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Classifica o prompt e define o modelo de destino apropriado.
        Retorna (target_model, tier_level, metadata).
        """
        lower_prompt = prompt.lower()
        matched_keywords = [kw for kw in self.ADVANCED_KEYWORDS if kw in lower_prompt]
        
        # Considera complexo se houver palavras-chave técnicas ou se o prompt for muito longo (>250 caracteres)
        is_complex = len(matched_keywords) > 0 or len(prompt) > 250
        
        if is_complex:
            target_model = settings.DEFAULT_ADVANCED_MODEL
            tier = "Tier-2 (Advanced & Reasoning)"
        else:
            target_model = settings.DEFAULT_FAST_MODEL
            tier = "Tier-1 (Fast & Economical)"

        metadata = {
            "matched_keywords": matched_keywords,
            "prompt_length": len(prompt),
            "complexity_flag": is_complex
        }
        
        logger.info(f"Routed prompt to '{target_model}' [{tier}] - Matched: {matched_keywords}")
        return target_model, tier, metadata


semantic_router = SemanticRouter()
