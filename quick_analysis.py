import json
import pandas as pd

with open('results/training_results.json', 'r') as f:
    results = json.load(f)

print("🎯 QUICK ANALYSIS SUMMARY")
print("="*60)

# Find best models
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
best_models = {}

for metric in metrics:
    best_score = 0
    best_model = None
    for model, res in results.items():
        if res['test'][metric] > best_score:
            best_score = res['test'][metric]
            best_model = model
    best_models[metric] = (best_model, best_score)

print("\n🏆 BEST MODELS BY METRIC (Test Set):")
for metric, (model, score) in best_models.items():
    print(f"   {metric.upper():12} → {model:20} ({score:.4f})")

# Check success criteria
print("\n✅ SUCCESS CRITERIA CHECK (≥92% Accuracy):")
passing_models = []
for model, res in results.items():
    acc = res['test']['accuracy']
    status = "✅ PASS" if acc >= 0.92 else "❌ FAIL"
    print(f"   {model:20} → {acc:.2%} {status}")
    if acc >= 0.92:
        passing_models.append(model)

print(f"\n📊 {len(passing_models)}/{len(results)} models meet success criteria")

# Training efficiency
print("\n⚡ FASTEST TRAINING:")
fastest = min(results.items(), key=lambda x: x[1]['training_time'])
print(f"   {fastest[0]} → {fastest[1]['training_time']:.2f}s")

print("\n💡 RECOMMENDATION:")
if passing_models:
    # Find best F1 among passing models
    best_passing = max(
        [(m, results[m]['test']['f1']) for m in passing_models],
        key=lambda x: x[1]
    )
    print(f"   Use '{best_passing[0]}' for deployment")
    print(f"   F1-Score: {best_passing[1]:.4f}")
else:
    print("   No model meets 92% accuracy requirement")
    print("   Consider: Feature engineering, hyperparameter tuning, or ensemble methods")