#!/usr/bin/env python3
"""
THE REALITY CHECK: Does Transformation Learning Scale to MNIST?

XOR/XNOR was 2D inputs, 4 examples, binary classification.
MNIST is 784D inputs, 60K examples, 10-class classification.

This is the CRITICAL test. If transformation learning fails here,
it's just a toy solution. If it WORKS here, it's revolutionary.

FORENSIC SETUP:
- Task 1: Classify digits 0-4 (5 classes)
- Task 2: Classify digits 5-9 (5 classes)
- Phase 1: Train base on 0-4 only
- Phase 2: Train transform on 5-9 only
- Evaluate: Both tasks should maintain >90% accuracy

CRITICAL QUESTIONS:
1. Can transform network handle 10D logit space (vs 2D)?
2. Does it generalize to real visual features?
3. Is this still parameter-efficient?
4. What happens to base task accuracy during Phase 2 training?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MNISTBaseNetwork(nn.Module):
    """Base network for Task 1 (digits 0-4)"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 classes (0-4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits for classes 0-4


class MNISTTransformNetwork(nn.Module):
    """Transform network to map Task 1 logits → Task 2 logits"""
    def __init__(self):
        super().__init__()
        # Input: 5D logits from base (classes 0-4)
        # Output: 5D logits for classes 5-9
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, base_logits):
        """Transform logits from digits 0-4 → logits for digits 5-9"""
        x = F.relu(self.fc1(base_logits))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_mnist_split():
    """Load MNIST and split into two tasks"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Split into Task 1 (0-4) and Task 2 (5-9)
    def filter_task1(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label < 5]
        return torch.utils.data.Subset(dataset, indices)

    def filter_task2(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label >= 5]
        subset = torch.utils.data.Subset(dataset, indices)
        # Remap labels 5-9 → 0-4
        return [(img, label - 5) for img, label in subset]

    task1_train = filter_task1(train_dataset)
    task1_test = filter_task1(test_dataset)

    # For task 2, we need to remap labels
    task2_train_indices = [i for i, (_, label) in enumerate(train_dataset) if label >= 5]
    task2_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label >= 5]

    task2_train = torch.utils.data.Subset(train_dataset, task2_train_indices)
    task2_test = torch.utils.data.Subset(test_dataset, task2_test_indices)

    return task1_train, task1_test, task2_train, task2_test


def remap_task2_labels(batch_labels):
    """Remap labels 5-9 → 0-4 for task 2"""
    return batch_labels - 5


def train_base_network(base_model, task1_train, num_epochs=5):
    """
    Phase 1: Train base network on digits 0-4 ONLY
    """
    print("\n" + "="*80)
    print("PHASE 1: Training Base Network on Digits 0-4")
    print("="*80)

    train_loader = torch.utils.data.DataLoader(
        task1_train, batch_size=128, shuffle=True
    )

    optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
    base_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = base_model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

        epoch_acc = 100. * correct / total
        print(f"\nEpoch {epoch+1} Complete - Accuracy: {epoch_acc:.2f}%")

    print("\nBase network training complete!")


def train_transform_network(base_model, transform_model, task2_train, num_epochs=5):
    """
    Phase 2: Train transform network on digits 5-9 ONLY
    Base network is FROZEN - no access to task 1 data!
    """
    print("\n" + "="*80)
    print("PHASE 2: Training Transform Network on Digits 5-9")
    print("Base network FROZEN - no access to digits 0-4!")
    print("="*80)

    # Freeze base
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    train_loader = torch.utils.data.DataLoader(
        task2_train, batch_size=128, shuffle=True
    )

    optimizer = torch.optim.Adam(transform_model.parameters(), lr=0.001)
    transform_model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # Remap labels 5-9 → 0-4
            target = remap_task2_labels(target)

            # Get frozen base network outputs
            with torch.no_grad():
                base_logits = base_model(data)

            # Transform to task 2 logits
            optimizer.zero_grad()
            task2_logits = transform_model(base_logits)
            loss = F.cross_entropy(task2_logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = task2_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

        epoch_acc = 100. * correct / total
        print(f"\nEpoch {epoch+1} Complete - Accuracy: {epoch_acc:.2f}%")

    print("\nTransform network training complete!")


def evaluate_both_tasks(base_model, transform_model, task1_test, task2_test):
    """
    Evaluate both tasks after training complete
    """
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    base_model.eval()
    transform_model.eval()

    # Evaluate Task 1 (digits 0-4)
    task1_loader = torch.utils.data.DataLoader(task1_test, batch_size=1000)
    task1_correct = 0
    task1_total = 0

    with torch.no_grad():
        for data, target in task1_loader:
            data, target = data.to(device), target.to(device)
            logits = base_model(data)
            pred = logits.argmax(dim=1)
            task1_correct += pred.eq(target).sum().item()
            task1_total += target.size(0)

    task1_acc = 100. * task1_correct / task1_total
    print(f"\nTask 1 (Digits 0-4) Accuracy: {task1_acc:.2f}% ({task1_correct}/{task1_total})")

    # Evaluate Task 2 (digits 5-9)
    task2_loader = torch.utils.data.DataLoader(task2_test, batch_size=1000)
    task2_correct = 0
    task2_total = 0

    with torch.no_grad():
        for data, target in task2_loader:
            data, target = data.to(device), target.to(device)
            target_remapped = remap_task2_labels(target)

            base_logits = base_model(data)
            task2_logits = transform_model(base_logits)
            pred = task2_logits.argmax(dim=1)
            task2_correct += pred.eq(target_remapped).sum().item()
            task2_total += target.size(0)

    task2_acc = 100. * task2_correct / task2_total
    print(f"Task 2 (Digits 5-9) Accuracy: {task2_acc:.2f}% ({task2_correct}/{task2_total})")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    if task1_acc >= 90 and task2_acc >= 90:
        print("✓ ✓ ✓ TRANSFORMATION LEARNING SCALES TO MNIST! ✓ ✓ ✓")
        print()
        print("This proves:")
        print("  ✓ Works on high-dimensional inputs (784D vs 2D)")
        print("  ✓ Works on real data (not just boolean functions)")
        print("  ✓ Works on multi-class (5 classes vs 2)")
        print("  ✓ Base task preserved (no catastrophic forgetting)")
        print()
        print("This IS a genuine breakthrough for continual learning!")
    elif task1_acc >= 90 and task2_acc >= 70:
        print("⚠ PARTIAL SUCCESS")
        print()
        print(f"Base task preserved ({task1_acc:.1f}%) but transform learning weaker ({task2_acc:.1f}%)")
        print("Transform network may need more capacity or training.")
    elif task1_acc < 90:
        print("✗ BASE TASK DEGRADED")
        print()
        print(f"Base task dropped to {task1_acc:.1f}% (should be >90%)")
        print("Frozen network not working as expected - investigate!")
    else:
        print("✗ TRANSFORMATION LEARNING FAILED ON REAL DATA")
        print()
        print(f"Task 2 accuracy: {task2_acc:.1f}% (needs >90%)")
        print("Transform approach doesn't scale beyond toy problems.")

    print("="*80)

    return task1_acc, task2_acc


def main():
    print("="*80)
    print("MNIST TRANSFORMATION LEARNING TEST")
    print("="*80)
    print()
    print("THE CRITICAL QUESTION:")
    print("  Does transformation learning scale beyond XOR/XNOR?")
    print()
    print("TEST SETUP:")
    print("  Task 1: Classify digits 0-4 (5 classes)")
    print("  Task 2: Classify digits 5-9 (5 classes)")
    print("  Method: Freeze base, train transform on task 2 only")
    print()
    print("SUCCESS CRITERIA:")
    print("  Both tasks >90% accuracy")
    print()
    print("="*80)

    # Load data
    print("\nLoading MNIST...")
    task1_train, task1_test, task2_train, task2_test = load_mnist_split()
    print(f"Task 1 (0-4): {len(task1_train)} train, {len(task1_test)} test")
    print(f"Task 2 (5-9): {len(task2_train)} train, {len(task2_test)} test")

    # Create models
    base_model = MNISTBaseNetwork().to(device)
    transform_model = MNISTTransformNetwork().to(device)

    # Count parameters
    base_params = sum(p.numel() for p in base_model.parameters())
    transform_params = sum(p.numel() for p in transform_model.parameters())
    print(f"\nBase network: {base_params:,} parameters")
    print(f"Transform network: {transform_params:,} parameters")
    print(f"Transform overhead: {100*transform_params/base_params:.1f}%")

    # Phase 1: Train base
    train_base_network(base_model, task1_train, num_epochs=5)

    # Phase 2: Train transform
    train_transform_network(base_model, transform_model, task2_train, num_epochs=5)

    # Evaluate both tasks
    task1_acc, task2_acc = evaluate_both_tasks(base_model, transform_model,
                                                task1_test, task2_test)

    return task1_acc, task2_acc


if __name__ == '__main__':
    main()
