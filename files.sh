
#!/usr/bin/env bash
set -euo pipefail

# Create directories
mkdir -p \
    configs/curricula \
    environments \
    training \
    evaluation \
    visualization \
    analysis \
    tests \
    results \
    checkpoints

# Configuration files
touch \
    configs/robot.yaml \
    configs/ppo.yaml \
    configs/experiment.yaml \
    configs/curricula/direct.yaml \
    configs/curricula/fixed.yaml \
    configs/curricula/adaptive.yaml \
    configs/curricula/shuffled.yaml

# Environment files
touch \
    environments/locomotion.py \
    environments/observations.py \
    environments/rewards.py \
    environments/perturbations.py \
    environments/curriculum.py

# Training files
touch \
    training/train.py \
    training/benchmark.py \
    training/checkpoint.py

# Evaluation files
touch \
    evaluation/evaluate.py \
    evaluation/push_recovery.py \
    evaluation/generalization.py \
    evaluation/metrics.py

# Visualization files
touch \
    visualization/render_policy.py \
    visualization/compare_policies.py

# Analysis files
touch \
    analysis/learning_curves.py \
    analysis/robustness_report.py \
    analysis/statistical_tests.py

# Project documentation
touch README.md

echo "Project structure created successfully!"
