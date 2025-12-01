# Code Generation Experiments

Transformation learning for continual learning on code generation tasks.

## Semantic Diversity Benchmark

**File**: [semantic_diversity_benchmark.py](semantic_diversity_benchmark.py)
**Results**: [results/semantic_diversity_results_seed42.json](results/semantic_diversity_results_seed42.json)

### Tasks

Three semantically diverse code generation tasks:

1. **Generation**: English description → Python code
2. **Summarization**: Python code → English description
3. **Translation**: Python code → JavaScript code

### Results Summary

| Approach | Avg Forgetting | Result |
|----------|----------------|--------|
| **Sequential Fine-Tuning** | **+36.87 BLEU** | Catastrophic forgetting |
| **Frozen Expert Routing** | **0.00 BLEU** | Zero forgetting (deterministic) |

### Dramatic Example

Summarization task after training:
- **Initially**: 100.00 BLEU ✅
- **After next task**: 0.13 BLEU ❌
- **Forgetting**: 99.87 BLEU

This is catastrophic forgetting in its purest form.

### Per-Task Breakdown

**Sequential Fine-Tuning**:
- Generation: 10.94 → 0.20 BLEU (+10.74 forgetting)
- Summarization: 100.00 → 0.13 BLEU (+99.87 forgetting) ⚠️
- Translation: 100.00 → 100.00 BLEU (0.00 - last task)

**Frozen Expert Routing**:
- Generation: 10.94 → 10.94 BLEU (0.00 forgetting) ✅
- Summarization: 100.00 → 100.00 BLEU (0.00 forgetting) ✅
- Translation: 100.00 → 100.00 BLEU (0.00 forgetting) ✅

## Quick Start

```bash
# Install dependencies
pip install transformers datasets evaluate sacrebleu torch

# Run benchmark
python semantic_diversity_benchmark.py --approach both

# Results saved to results/semantic_diversity_results_seed42.json
```

## Architecture

**Frozen Expert Routing**:
```python
# Train experts independently
expert_gen = CodeT5Expert().train(generation_data)
expert_gen.freeze()  # Parameters locked

expert_sum = CodeT5Expert().train(summarization_data)
expert_sum.freeze()

expert_trans = CodeT5Expert().train(translation_data)
expert_trans.freeze()

# Route inputs to correct expert
router = train_router([expert_gen, expert_sum, expert_trans])

# Zero forgetting guaranteed (frozen params can't change)
```

## Key Insight

**Semantic diversity** (different task objectives) causes severe catastrophic forgetting in sequential learning, but frozen expert routing provides a deterministic zero-forgetting guarantee through parameter isolation.
