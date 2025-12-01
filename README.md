# Transformation Learning for Continual Learning

> What if catastrophic forgetting isn't about memory, but about asking neural networks to solve impossible problems?

## The Problem

In continual learning, neural networks are expected to learn new tasks sequentially without forgetting previous ones. The field has struggled with **catastrophic forgetting** for decades—train on Task 2, and performance on Task 1 collapses to near-zero.

The standard explanation: "The network forgets what it learned before."

But what if that's wrong? What if the network is being asked to do something mathematically impossible?

## The Key Insight

Consider learning XOR then XNOR on the same inputs:
```
XOR:  [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
XNOR: [0,0]→1, [0,1]→0, [1,0]→0, [1,1]→1
```

For input `[0,0]`, the network must output both `0` (XOR) and `1` (XNOR). **This is a deterministic contradiction**, not a memory problem.

Traditional approach: Force the network to learn conflicting mappings `X→Y₁` and `X→Y₂`
**Transformation learning**: Learn task relationships `X→Y₁` and `Y₁→Y₂`

Instead of overwriting, we **transform** the output space.

## Cross-Domain Results

| Domain | Tasks | Approach | Result | Forgetting |
|--------|-------|----------|--------|------------|
| **Vision** | 5 (MNIST) | Star topology | 98.3% acc | 0% (improved!) |
| **Code** | 3 (Gen/Sum/Trans) | Frozen experts | 100 BLEU | 0.00 BLEU |

Both domains achieve **deterministic zero forgetting** through architectural solutions, not regularization tricks.

---

## Vision Domain (MNIST)

### Results Summary

| Experiment | Accuracy | Key Finding |
|------------|----------|-------------|
| XOR/XNOR (baseline) | 0% | Complete catastrophic forgetting |
| XOR/XNOR (transformation) | 100% | Perfect continual learning |
| **MNIST 2-task (logits)** | 80.6% | Logit-level transforms insufficient |
| **MNIST 2-task (features)** | **96.9%** | **Feature-level transforms work!** |
| **MNIST 5-task (star topology)** | **98.3%** | **Scales to N>2 tasks** |
| **Parameter savings** | **75.6%** | **vs. 5 separate networks** |

**Surprising discovery**: The base task IMPROVED (99.86% → 99.91%) after training 4 additional tasks. No catastrophic forgetting—the opposite!

### Architecture: Star Topology
```
    Base Network (frozen)
          ↓ features (128D)
    ┌─────┼─────┬─────┐
    ↓     ↓     ↓     ↓
  Task2 Task3 Task4 Task5
```

All transforms learn from the same frozen base features. No error accumulation through chains.

---

## Code Generation Domain

### Results Summary

| Approach | Avg Forgetting | Result |
|----------|----------------|--------|
| **Sequential Fine-Tuning** | **+36.87 BLEU** | Catastrophic forgetting |
| **Frozen Expert Routing** | **0.00 BLEU** | Zero forgetting (deterministic) |

### Dramatic Example

Summarization task:
- After training: **100.00 BLEU** ✅
- After next task: **0.13 BLEU** ❌
- **Forgetting**: 99.87 BLEU

This is catastrophic forgetting in its purest form.

### Architecture: Frozen Expert Routing

```python
# Train experts independently
expert_gen = CodeT5Expert().train(generation_data)
expert_gen.freeze()  # Parameters locked

expert_sum = CodeT5Expert().train(summarization_data)
expert_sum.freeze()

expert_trans = CodeT5Expert().train(translation_data)
expert_trans.freeze()

# Route inputs to correct expert
router = LSTM_Router([expert_gen, expert_sum, expert_trans])

# Zero forgetting guaranteed (frozen params can't change)
```

---

## Quick Start

### Vision Experiments (MNIST)

Each experiment takes ~2-5 minutes on CPU, ~30 seconds on GPU.

```bash
# Install dependencies
pip install torch torchvision numpy

# Run vision experiments
cd experiments/vision/
python mnist_n_task_scaling.py
```

### Code Generation Experiments

Experiments take ~10-15 minutes on GPU.

```bash
# Install dependencies
pip install transformers datasets evaluate sacrebleu torch

# Run code generation benchmark
cd experiments/code/
python semantic_diversity_benchmark.py --approach both
```

---

## Key Findings

### 1. Transformation Learning Works Across Domains

- **Vision (MNIST)**: 98.3% accuracy on 5 tasks
- **Code (Gen/Sum/Trans)**: 0.00 BLEU forgetting on 3 tasks
- Reformulates continual learning as learning task relationships
- Achieves deterministic zero-forgetting guarantees

### 2. Architecture Matters More Than Regularization

**Vision**: Star topology prevents error accumulation
**Code**: Frozen experts provide mathematical guarantee

Both use **architectural isolation** instead of regularization penalties (EWC, SI, etc.)

### 3. Representation Level Matters (Vision)

- **Logit-level** transforms (5D): 80.6% accuracy
- **Feature-level** transforms (128D): **96.9% accuracy** (+16%)

Why? Intermediate features contain general patterns (edges, textures) that transfer better than task-specific outputs.

### 4. Accidentally Solved Loss of Plasticity (Vision)

By freezing the base network, we prevent gradient-induced degradation (dormant neurons, weight inflation). The base task actually improves as we add more tasks!

### 5. Router Independence (Code)

Zero-forgetting achieved even when router fails to converge. The deterministic guarantee is **independent of router quality**.

---

## What Makes This Different

**vs. EWC/SI/PackNet**: Those try to protect old knowledge while learning new tasks. This reformulates the problem—no conflict means no forgetting.

**vs. Progressive Neural Networks**: Similar idea (separate columns), but they grow linearly with tasks. Vision domain achieves 75% parameter savings through shared frozen features. Code domain has no savings but deterministic guarantees.

**vs. Task-Incremental Learning**: Most methods assume task IDs available. This work proves transformation learning scales when IDs are available, and explores routing with minimal supervision (vision: 79.7% without IDs).

---

## Honest Limitations

### Vision Domain
1. **Task IDs required at inference** (98.3%) or routing (79.7%)
2. **Star topology** requires one "base" task
3. **Tested on MNIST** - CIFAR-100 and ImageNet remain open

### Code Domain
1. **No parameter savings** - Separate expert per task
2. **Router instability** - 43% seed success rate
3. **Task IDs required** - Router provides task detection
4. **Tested on CodeT5-small** - Larger models remain open

---

## Repository Structure

```
.
├── README.md                          # You are here
├── experiments/
│   ├── vision/                        # MNIST experiments (4 files)
│   │   ├── mnist_transformation_test.py      # Initial test (80.6%)
│   │   ├── mnist_larger_transform.py         # Breakthrough (96.9%)
│   │   ├── mnist_n_task_scaling.py           # N=5 scaling (98.3%)
│   │   ├── mnist_reward_routing.py           # Task routing (79.7%)
│   │   └── README.md
│   ├── code/                          # Code generation experiments
│   │   ├── semantic_diversity_benchmark.py
│   │   ├── results/
│   │   └── README.md
│   └── README.md
├── docs/
│   └── CROSS_DOMAIN_RESULTS.md        # Detailed cross-domain analysis
├── archive/                           # All XOR/XNOR experiments, C code
├── INVESTIGATION_SUMMARY.md           # Complete investigation (50+ vision experiments)
├── FINAL_FINDINGS.md                  # Publication-ready findings (vision)
└── data/                              # MNIST dataset (auto-downloaded)
```

---

## Read More

- [experiments/README.md](experiments/README.md) - Overview of all experiments
- [experiments/vision/README.md](experiments/vision/README.md) - Vision experiments details
- [experiments/code/README.md](experiments/code/README.md) - Code experiments details
- [docs/CROSS_DOMAIN_RESULTS.md](docs/CROSS_DOMAIN_RESULTS.md) - Complete cross-domain analysis
- [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md) - Vision investigation (50+ experiments)
- [FINAL_FINDINGS.md](FINAL_FINDINGS.md) - Publication-ready findings

---

## Citation

If you find this work useful:

```bibtex
@misc{transformation_learning_2025,
  title={Transformation Learning: Cross-Domain Continual Learning Without Forgetting},
  author={Your Name},
  year={2025},
  note={GitHub: https://github.com/VoidTactician/transformation-learning}
}
```

---

## License

MIT - Use freely, cite if helpful

---

**Investigation completed**: November 6, 2025 (vision), December 1, 2025 (code)
**Total experiments**: 50+ (vision), 1 (code)
**Key discovery**: Catastrophic forgetting is a representation problem, not a memory problem
**Domains validated**: Vision (MNIST), Code (Gen/Sum/Trans)
