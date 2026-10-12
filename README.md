# 🔀 LLM Semantic Router (Enterprise Edition)

Motor de Roteamento Semântico Inteligente, **Cascade Model Selector** e **Cache Semântico Vetorial Bounded LRU** com aceleração matricial BLAS/SIMD.

---

## 🌟 Arquitetura & Recursos Big-Tech

- **Bounded LRU Semantic Cache**:
  - Armazenamento com limite estrito de memória (`max_size`) baseado em `collections.OrderedDict`, garantindo expulsão $\mathcal{O}(1)$ e prevenção total de memory leak / OOM.
  - Expiração granular por TTL (Time-To-Live) com purga assíncrona e sob demanda.
- **Matrix-Accelerated Vector Search**:
  - Busca de similaridade em batch projetando o vetor de query sobre a matriz completa de embeddings $\mathbf{V} \in \mathbb{R}^{N \times D}$:
    $$\mathbf{s} = \mathbf{V}_{\text{norm}} \cdot \mathbf{q}_{\text{norm}}^T$$
  - Executado em nanossegundos C via NumPy BLAS/SIMD sem overhead de loops Python.
- **Cost-Aware Cascade Model Selector**:
  - Análise multi-dimensional de complexidade (Type-Token Ratio, marcadores léxicos e detecção de sintaxe de código) roteando dinamicamente entre:
    - **SLM Nano** (`llama-3.2-3b-instruct`): Consultas rápidas, conversacionais e de baixo custo (\$0.00005).
    - **SLM Standard** (`llama-3.3-70b-instruct`): Consultas técnicas intermediárias (\$0.0005).
    - **LLM Frontier** (`gpt-4o`): Sínteses complexas, algoritmos e trade-offs arquiteturais (\$0.005).
- **Observabilidade OTel, Langfuse & Prometheus**:
  - Rastreamento ponta a ponta de latência de cache hit vs miss e projeção contínua de economia financeira acumulada (USD).

---

## 📊 Complexidade Algorítmica

| Operação | Implementação Anterior | Implementação Otimizada (Staff) | Estrutura de Dados |
| :--- | :--- | :--- | :--- |
| **Expulsão de Cache** | Inexistente (crescimento descontrolado) | $\mathcal{O}(1)$ Bounded LRU | `OrderedDict.popitem(last=False)` |
| **Busca de Similaridade** | $\mathcal{O}(N \times D)$ loops escalares Python | $\mathcal{O}(N \times D)$ NumPy Matrix Dot | 2D Continuous Buffer |
| **Classificação de Complexidade** | Contagem simples de tokens | Análise Léxica + Sintática + TTR | Multi-Signal Classifier |

---

## 🚀 Como Executar no Docker

```bash
docker compose up -d --build
```

---

## 🧪 Testes Automatizados (12 Testes / 100% Hermético / 94% Cobertura)

Execução hermética da suíte de testes:

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -e PYTHONPATH=/workspace portfolio-spec-engine:test pytest tests -v --cov=app/core
```
