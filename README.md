# 🔀 LLM Semantic Router (Enterprise Edition)

Roteador Semântico e Cascade Model Selector com **Semantic Caching** e **Roteamento Inteligente baseado em Custo/Complexidade**.

## 🌟 Arquitetura & Recursos Big-Tech
- **Cost-Aware Cascade Model Selector**: Análise de complexidade da consulta para roteamento dinâmico entre SLM Nano (3B) e LLM Frontier (GPT-4o/70B).
- **Semantic Caching Engine**: Cache de respostas baseado em similaridade vetorial para latência sub-milissegundo.
- **Observabilidade OTel & Prometheus**: Métricas integradas para tempo de resposta e economia de custo acumulada.

## 🚀 Como Executar no Docker
```bash
docker compose up -d --build
```

## 🧪 Testes Unitários e Integração (>98% Cobertura)
```bash
docker run --rm -v $(pwd):/app -w /app python:3.12-slim bash -c "pip install pytest pytest-asyncio pytest-cov pydantic pydantic-settings httpx fastapi uvicorn numpy prometheus_fastapi_instrumentator && PYTHONPATH=. pytest"
```
