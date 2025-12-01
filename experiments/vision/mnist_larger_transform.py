#!/usr/bin/env python3
"""
HYPOTHESIS TEST: Is transform network too small?

Previous result: 80.6% with 1.4K parameter transform
Test: Increase transform capacity significantly

VARIANTS TO TEST:
1. Larger MLP: 5→128→128→64→5 (~17K params, 12x larger)
2. Very Deep: 5→64→64→64→64→5 (~21K params, many layers)
3. Feature transform: Map intermediate features, not just logits

If larger transform reaches >90%, capacity was the issue.
If still <90%, it's a fundamental limitation of the approach.
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
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        features = F.relu(self.fc1(x))
        logits = self.fc2(features)

        if return_features:
            return logits, features
        return logits


class LargeTransformNetwork(nn.Module):
    """LARGER transform network - test capacity hypothesis"""
    def __init__(self):
        super().__init__()
        # Much larger: 5→128→128→64→5
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, base_logits):
        x = F.relu(self.fc1(base_logits))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class VeryDeepTransform(nn.Module):
    """Very deep transform - test depth hypothesis"""
    def __init__(self):
        super().__init__()
        # 5 layers: 5→64→64→64→64→5
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 5)
        self.dropout = nn.Dropout(0.1)

    def forward(self, base_logits):
        x = F.relu(self.fc1(base_logits))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class FeatureTransform(nn.Module):
    """Transform from intermediate features instead of just logits"""
    def __init__(self):
        super().__init__()
        # Input: 128D features from fc1 of base
        # Output: 5D logits for task 2
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


def load_mnist_split():
    """Load MNIST and split into two tasks"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    def filter_task1(dataset):
        indices = [i for i, (_, label) in enumerate(dataset) if label < 5]
        return torch.utils.data.Subset(dataset, indices)

    task1_train = filter_task1(train_dataset)
    task1_test = filter_task1(test_dataset)

    task2_train_indices = [i for i, (_, label) in enumerate(train_dataset) if label >= 5]
    task2_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label >= 5]

    task2_train = torch.utils.data.Subset(train_dataset, task2_train_indices)
    task2_test = torch.utils.data.Subset(test_dataset, task2_test_indices)

    return task1_train, task1_test, task2_train, task2_test


def remap_task2_labels(batch_labels):
    return batch_labels - 5


def train_base_network(base_model, task1_train, num_epochs=5):
    """Train base on digits 0-4"""
    print("\nTraining base network...")
    train_loader = torch.utils.data.DataLoader(task1_train, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=0.001)
    base_model.train()

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logits = base_model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        print(f"  Epoch {epoch+1}: {100.*correct/total:.2f}%")

    print("Base training complete!")


def train_transform_logits(base_model, transform_model, task2_train, num_epochs=10):
    """Train transform on logits (original approach)"""
    print("\nTraining transform network (on logits)...")

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    train_loader = torch.utils.data.DataLoader(task2_train, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(transform_model.parameters(), lr=0.001)
    transform_model.train()

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = remap_task2_labels(target)

            with torch.no_grad():
                base_logits = base_model(data)

            optimizer.zero_grad()
            task2_logits = transform_model(base_logits)
            loss = F.cross_entropy(task2_logits, target)
            loss.backward()
            optimizer.step()

            pred = task2_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        print(f"  Epoch {epoch+1}: {100.*correct/total:.2f}%")

    print("Transform training complete!")


def train_transform_features(base_model, transform_model, task2_train, num_epochs=10):
    """Train transform on features (new approach)"""
    print("\nTraining transform network (on features)...")

    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()

    train_loader = torch.utils.data.DataLoader(task2_train, batch_size=128, shuffle=True)
    optimizer = torch.optim.Adam(transform_model.parameters(), lr=0.001)
    transform_model.train()

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = remap_task2_labels(target)

            with torch.no_grad():
                _, features = base_model(data, return_features=True)

            optimizer.zero_grad()
            task2_logits = transform_model(features)
            loss = F.cross_entropy(task2_logits, target)
            loss.backward()
            optimizer.step()

            pred = task2_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        print(f"  Epoch {epoch+1}: {100.*correct/total:.2f}%")

    print("Transform training complete!")


def evaluate_logits(base_model, transform_model, task1_test, task2_test):
    """Evaluate using logit transform"""
    base_model.eval()
    transform_model.eval()

    # Task 1
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

    # Task 2
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
    return task1_acc, task2_acc


def evaluate_features(base_model, transform_model, task1_test, task2_test):
    """Evaluate using feature transform"""
    base_model.eval()
    transform_model.eval()

    # Task 1
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

    # Task 2
    task2_loader = torch.utils.data.DataLoader(task2_test, batch_size=1000)
    task2_correct = 0
    task2_total = 0

    with torch.no_grad():
        for data, target in task2_loader:
            data, target = data.to(device), target.to(device)
            target_remapped = remap_task2_labels(target)
            _, features = base_model(data, return_features=True)
            task2_logits = transform_model(features)
            pred = task2_logits.argmax(dim=1)
            task2_correct += pred.eq(target_remapped).sum().item()
            task2_total += target.size(0)

    task2_acc = 100. * task2_correct / task2_total
    return task1_acc, task2_acc


def main():
    print("="*80)
    print("CAPACITY TEST: Does Larger Transform Network Help?")
    print("="*80)

    task1_train, task1_test, task2_train, task2_test = load_mnist_split()

    results = []

    # Test 1: Large MLP transform
    print("\n" + "="*80)
    print("TEST 1: Large MLP Transform (5→128→128→64→5)")
    print("="*80)

    base1 = MNISTBaseNetwork().to(device)
    transform1 = LargeTransformNetwork().to(device)
    params1 = sum(p.numel() for p in transform1.parameters())
    print(f"Transform parameters: {params1:,}")

    train_base_network(base1, task1_train, num_epochs=5)
    train_transform_logits(base1, transform1, task2_train, num_epochs=10)
    t1_acc, t2_acc = evaluate_logits(base1, transform1, task1_test, task2_test)

    print(f"\nResults: Task1={t1_acc:.2f}%, Task2={t2_acc:.2f}%")
    results.append(("Large MLP", params1, t1_acc, t2_acc))

    # Test 2: Very Deep transform
    print("\n" + "="*80)
    print("TEST 2: Very Deep Transform (5 layers)")
    print("="*80)

    base2 = MNISTBaseNetwork().to(device)
    transform2 = VeryDeepTransform().to(device)
    params2 = sum(p.numel() for p in transform2.parameters())
    print(f"Transform parameters: {params2:,}")

    train_base_network(base2, task1_train, num_epochs=5)
    train_transform_logits(base2, transform2, task2_train, num_epochs=10)
    t1_acc, t2_acc = evaluate_logits(base2, transform2, task1_test, task2_test)

    print(f"\nResults: Task1={t1_acc:.2f}%, Task2={t2_acc:.2f}%")
    results.append(("Very Deep", params2, t1_acc, t2_acc))

    # Test 3: Feature transform
    print("\n" + "="*80)
    print("TEST 3: Feature Transform (128D features → 5D logits)")
    print("="*80)

    base3 = MNISTBaseNetwork().to(device)
    transform3 = FeatureTransform().to(device)
    params3 = sum(p.numel() for p in transform3.parameters())
    print(f"Transform parameters: {params3:,}")

    train_base_network(base3, task1_train, num_epochs=5)
    train_transform_features(base3, transform3, task2_train, num_epochs=10)
    t1_acc, t2_acc = evaluate_features(base3, transform3, task1_test, task2_test)

    print(f"\nResults: Task1={t1_acc:.2f}%, Task2={t2_acc:.2f}%")
    results.append(("Feature Transform", params3, t1_acc, t2_acc))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Approach':<20} {'Params':>10} {'Task1':>8} {'Task2':>8} {'Success':>10}")
    print("-"*80)
    print(f"{'Baseline (small)':<20} {'1,413':>10} {'99.0':>8} {'80.6':>8} {'No':>10}")
    for name, params, t1, t2 in results:
        success = "Yes" if (t1 >= 90 and t2 >= 90) else "No"
        print(f"{name:<20} {params:>10,} {t1:>8.2f} {t2:>8.2f} {success:>10}")

    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    best_t2 = max(results, key=lambda x: x[3])
    if best_t2[3] >= 90:
        print(f"✓ SUCCESS: {best_t2[0]} achieved {best_t2[3]:.1f}%!")
        print("\nCapacity was the bottleneck. Larger transform network solves it.")
    else:
        print(f"✗ LIMITATION: Best result {best_t2[3]:.1f}% ({best_t2[0]})")
        print("\nEven with 10x more parameters, can't reach 90%.")
        print("This suggests a fundamental limitation:")
        print("  - Logits optimized for 0-4 don't transfer well to 5-9")
        print("  - May need shared representation learning")
        print("  - Or different task decomposition approach")

    print("="*80)


if __name__ == '__main__':
    main()
