import pytest
from app.core.cost_cascade_router import CostAwareCascadeRouter, ModelTier, cascade_router


def test_cost_cascade_router_simple_query():
    decision = cascade_router.route_query("Qual eh a capital da Franca?")
    assert decision.selected_tier == ModelTier.SLM_NANO
    assert decision.target_model_name == "llama-3.2-3b-instruct"


def test_cost_cascade_router_complex_query():
    decision = cascade_router.route_query("Analise detalhadamente os trade-offs e a arquitetura de sistemas distribuídos sob alto tráfego.")
    assert decision.selected_tier == ModelTier.LLM_FRONTIER
    assert decision.target_model_name == "gpt-4o"
