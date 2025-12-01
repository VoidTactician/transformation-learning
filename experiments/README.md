# Experiments

This directory contains transformation learning experiments across two domains: **vision** and **code generation**.

## Structure

```
experiments/
├── vision/              # MNIST image classification (5 tasks)
│   ├── mnist_transformation_test.py
│   ├── mnist_larger_transform.py
│   ├── mnist_n_task_scaling.py
│   ├── mnist_reward_routing.py
│   └── README.md
│
└── code/                # Code generation (3 tasks)
    ├── semantic_diversity_benchmark.py
    ├── results/
    └── README.md
```

## Cross-Domain Results

| Domain | Tasks | Approach | Accuracy/BLEU | Forgetting |
|--------|-------|----------|---------------|------------|
| **Vision** | 5 (MNIST digits) | Star topology | 98.3% | 0% (improved!) |
| **Code** | 3 (Gen/Sum/Trans) | Frozen experts | 100.00 BLEU | 0.00 BLEU |

## Key Finding

Transformation learning prevents catastrophic forgetting across different domains:
- **Vision**: Feature-level transforms with star topology
- **Code**: Frozen expert routing with parameter isolation

Both approaches achieve **zero catastrophic forgetting** through architectural solutions rather than regularization tricks.

## Running Experiments

### Vision (MNIST)
```bash
cd vision/
python mnist_n_task_scaling.py  # ~2-5 min on CPU
```

### Code Generation
```bash
cd code/
pip install transformers datasets evaluate sacrebleu
python semantic_diversity_benchmark.py --approach both  # ~10-15 min on GPU
```

## More Information

- [Vision experiments README](vision/README.md)
- [Code experiments README](code/README.md)
- [Main README](../README.md)
