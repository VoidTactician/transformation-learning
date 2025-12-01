# Vision Experiments (MNIST)

Transformation learning for continual learning on vision tasks.

## Experiments

### 1. Initial Test: Logit-Level Transforms
**File**: [mnist_transformation_test.py](mnist_transformation_test.py)
- **Result**: 80.6% accuracy
- **Finding**: Logit-level transforms (5D) insufficient

### 2. Breakthrough: Feature-Level Transforms
**File**: [mnist_larger_transform.py](mnist_larger_transform.py)
- **Result**: 96.9% accuracy (+16% improvement)
- **Key Insight**: Feature-level transforms (128D) work better than logit-level (5D)

### 3. N-Task Scaling: 5 Tasks with Star Topology
**File**: [mnist_n_task_scaling.py](mnist_n_task_scaling.py)
- **Result**: 98.3% accuracy on 5 tasks
- **Finding**: Star topology prevents error accumulation
- **Parameter savings**: 75.6% vs 5 separate networks

### 4. Reward-Based Routing: Task Detection
**File**: [mnist_reward_routing.py](mnist_reward_routing.py)
- **Result**: 79.7% accuracy (without task IDs)
- **Comparison**: 98.3% with task IDs, 79.7% with routing

## Quick Start

Each experiment runs in ~2-5 minutes on CPU, ~30 seconds on GPU.

```bash
# Install dependencies
pip install torch torchvision numpy

# Run any experiment
python mnist_n_task_scaling.py
```

## Key Findings

**Star Topology Architecture**:
```
    Base Network (frozen)
          ↓ features
    ┌─────┼─────┬─────┐
    ↓     ↓     ↓     ↓
  Task2 Task3 Task4 Task5
```

All transforms learn from the same frozen base features - no error accumulation.

**Surprising Result**: The base task actually IMPROVED (99.86% → 99.91%) after training 4 additional tasks. No catastrophic forgetting - the opposite!
