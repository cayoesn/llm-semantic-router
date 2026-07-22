from enum import StrEnum
from pydantic import BaseModel, Field


class ModelTier(StrEnum):
    SLM_NANO = "SLM_NANO"  # 1B-3B
    SLM_STANDARD = "SLM_STANDARD"  # 8B
    LLM_FRONTIER = "LLM_FRONTIER"  # 70B+ / Claude / GPT-4o


class CascadeRoutingDecision(BaseModel):
    query: str
    selected_tier: ModelTier
    estimated_cost_usd: float
    complexity_score: float = Field(ge=0.0, le=1.0)
    target_model_name: str
    reasoning: str


class CostAwareCascadeRouter:
    """Enterprise Cost-Aware Model Cascade & Semantic Router Engine.
    
    Analisa a complexidade semântica da consulta para rotear dinamicamente para o
    modelo com melhor relação custo/benefício (SLM Nano x SLM Standard x LLM Frontier).
    """

    COMPLEXITY_INDICATORS = [
        "analise detalhadamente", "compare e contraste", "escreva um algoritmo",
        "prove o teorema", "refatore o código", "arquitetura de sistemas",
        "trade-offs", "diagnostique o erro", "explique a física quântica"
    ]

    def route_query(self, query: str) -> CascadeRoutingDecision:
        lower = query.lower()
        complexity_hits = sum(1 for kw in self.COMPLEXITY_INDICATORS if kw in lower)
        length_factor = min(0.5, len(query) / 1000.0)
        
        complexity_score = min(1.0, (complexity_hits * 0.25) + length_factor)

        if complexity_score < 0.25:
            tier = ModelTier.SLM_NANO
            model = "llama-3.2-3b-instruct"
            cost = 0.00005
            reason = "Consulta simples roteada para SLM Nano para resposta ultrarrápida e baixo custo."
        elif complexity_score < 0.6:
            tier = ModelTier.SLM_STANDARD
            model = "llama-3.3-70b-instruct"
            cost = 0.0005
            reason = "Consulta de complexidade média roteada para SLM Standard."
        else:
            tier = ModelTier.LLM_FRONTIER
            model = "gpt-4o"
            cost = 0.005
            reason = "Consulta altamente complexa/técnica roteada para LLM Frontier."

        return CascadeRoutingDecision(
            query=query,
            selected_tier=tier,
            estimated_cost_usd=cost,
            complexity_score=round(complexity_score, 2),
            target_model_name=model,
            reasoning=reason,
        )


# Instância Singleton
cascade_router = CostAwareCascadeRouter()
