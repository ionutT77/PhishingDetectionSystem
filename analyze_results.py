import json
import pandas as pd
from pathlib import Path

# Load training results
with open('results/training_results.json', 'r') as f:
    results = json.load(f)

# Display detailed metrics for each model
for model_name, metrics in results.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    print(f"\nTraining Time: {metrics['training_time']:.2f}s")
    
    print("\n📊 Validation Set:")
    for metric, value in metrics['validation'].items():
        if metric != 'confusion_matrix':
            print(f"   {metric}: {value:.4f}")
    
    print("\n📊 Test Set:")
    for metric, value in metrics['test'].items():
        if metric != 'confusion_matrix':
            print(f"   {metric}: {value:.4f}")
    
    # Confusion Matrix
    cm = metrics['test']['confusion_matrix']
    print("\n🔢 Test Confusion Matrix:")
    print(f"   TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"   FN: {cm[1][0]}, TP: {cm[1][1]}")