import re
import logging
from typing import Dict, Any, Tuple
from app.config import settings

logger = logging.getLogger("llm_semantic_router.router")


class SemanticRouter:
    """
    Semantic Intent Router that evaluates prompt complexity to select
    the appropriate LLM tier (Fast Flash vs. Advanced Pro).
    """

    # Keywords indicating high technical complexity or deep reasoning
    ADVANCED_KEYWORDS = [
        "code", "refactor", "architecture", "algorithm", "optimize",
        "math", "analysis", "debug", "sql", "python", "docker",
        "fastapi", "microservices", "proof", "prove", "reasoning"
    ]

    def classify_and_route(self, prompt: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Classifies the prompt and selects the appropriate target model.
        Returns (target_model, tier_level, metadata).
        """
        lower_prompt = prompt.lower()
        matched_keywords = [kw for kw in self.ADVANCED_KEYWORDS if kw in lower_prompt]

        # Treat as complex if technical keywords match or if the prompt exceeds 250 characters
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
