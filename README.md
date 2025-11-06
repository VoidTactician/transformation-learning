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

## Results Summary

| Experiment | Accuracy | Key Finding |
|------------|----------|-------------|
| XOR/XNOR (baseline) | 0% | Complete catastrophic forgetting |
| XOR/XNOR (transformation) | 100% | Perfect continual learning |
| **MNIST 2-task (logits)** | 80.6% | Logit-level transforms insufficient |
| **MNIST 2-task (features)** | **96.9%** | **Feature-level transforms work!** |
| **MNIST 5-task (star topology)** | **98.3%** | **Scales to N>2 tasks** |
| **Parameter savings** | **75.6%** | **vs. 5 separate networks** |

**Surprising discovery**: The base task IMPROVED (99.86% → 99.91%) after training 4 additional tasks. No catastrophic forgetting—the opposite!

## Quick Start

### Installation

```bash
# Create environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install torch torchvision numpy
```

### Run Experiments

Each experiment takes ~2-5 minutes on CPU, ~30 seconds on GPU.

```bash
# 1. Initial test: Logit-level transforms (80.6%)
python experiments/mnist_transformation_test.py

# 2. Breakthrough: Feature-level transforms (96.9%)
python experiments/mnist_larger_transform.py

# 3. N-task scaling: 5 tasks with star topology (98.3%)
python experiments/mnist_n_task_scaling.py

# 4. Reward-based routing: Task detection (79.7%)
python experiments/mnist_reward_routing.py
```

## Key Findings

### 1. Transformation Learning Scales to Real Data
- **100% on XOR/XNOR** (toy problem) → **98.3% on MNIST** (real data, 5 tasks)
- Reformulates continual learning as learning task relationships
- Requires task IDs at inference time
- **Critical insight**: Transform intermediate features (128D), not output logits (5D)

### 2. Star Topology Prevents Error Accumulation
Instead of chaining (Task1 → Task2 → Task3 → ...), we use a star:
```
    Base Network (frozen)
          ↓ features
    ┌─────┼─────┬─────┐
    ↓     ↓     ↓     ↓
  Task2 Task3 Task4 Task5
```
All transforms learn from the same frozen base features. No error accumulation through chains.

### 3. Representation Level Matters
- **Logit-level** transforms (5D): 80.6% accuracy
- **Feature-level** transforms (128D): **96.9% accuracy** (+16%)

Why? Intermediate features contain general visual patterns (edges, curves, textures) that transfer better than task-specific logits.

### 4. Accidentally Solved Loss of Plasticity
By freezing the base network, we prevent gradient-induced degradation (dormant neurons, weight inflation, rank collapse). The base task actually improves as we add more tasks!

## What Makes This Different

**vs. EWC/SI/PackNet**: Those try to protect old knowledge while learning new tasks. This reformulates the problem—no conflict means no forgetting.

**vs. Progressive Neural Networks**: Similar idea (separate columns), but they grow linearly with tasks. Transformation learning uses shared frozen features (75% parameter savings).

**vs. Task-Incremental Learning**: Most methods assume task IDs available. This work proves transformation learning scales when IDs are available, and explores routing with minimal supervision (binary feedback).

## Honest Limitations

1. **Task IDs required at inference** - You need to know which task you're solving
2. **Reward-based routing** (no task IDs) achieves 79.7% vs 98.3% (with IDs)
3. **Tested on MNIST** - CIFAR-100 and real-world benchmarks remain open questions
4. **Star topology** requires one "base" task - what if there's no natural base?

## Read More

- [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md) - Complete investigation (50+ experiments)
- [FINAL_FINDINGS.md](FINAL_FINDINGS.md) - Publication-ready findings
- [archive/](archive/) - All experimental code from the investigation

## Repository Structure

```
.
├── README.md                    # You are here
├── INVESTIGATION_SUMMARY.md     # Complete forensic investigation
├── FINAL_FINDINGS.md            # Publication-ready summary
├── experiments/                 # 4 key MNIST experiments
│   ├── mnist_transformation_test.py      # Initial test (80.6%)
│   ├── mnist_larger_transform.py         # Breakthrough (96.9%)
│   ├── mnist_n_task_scaling.py           # N=5 scaling (98.3%)
│   └── mnist_reward_routing.py           # Task routing (79.7%)
├── archive/                     # All XOR/XNOR experiments, docs, C code
└── data/                        # MNIST dataset (auto-downloaded)
```

## Citation

If you find this work useful:

```bibtex
@misc{transformation_learning_2025,
  title={Transformation Learning: Reformulating Continual Learning as Task Relationship Learning},
  author={Your Name},
  year={2025},
  note={GitHub: https://github.com/your-username/transformation-learning}
}
```

## License

MIT - Use freely, cite if helpful

---

**Investigation completed**: November 6, 2025
**Total experiments**: 50+
**Key discovery**: Catastrophic forgetting is a representation problem, not a memory problem
