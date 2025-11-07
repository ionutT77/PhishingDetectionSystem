import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load results
with open('results/training_results.json', 'r') as f:
    results = json.load(f)

# Create output directory
Path('results/visualizations').mkdir(parents=True, exist_ok=True)

# 1. Compare all metrics across models
metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
models = list(results.keys())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Model Performance Comparison (Test Set)', fontsize=16, fontweight='bold')

for idx, metric in enumerate(metrics_names):
    ax = axes[idx // 3, idx % 3]
    
    values = [results[model]['test'][metric] for model in models]
    
    bars = ax.barh(models, values, color=sns.color_palette("husl", len(models)))
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')

# Training time comparison
ax = axes[1, 2]
times = [results[model]['training_time'] for model in models]
bars = ax.barh(models, times, color=sns.color_palette("husl", len(models)))
ax.set_xlabel('Training Time (seconds)')
for i, v in enumerate(times):
    ax.text(v + 0.1, i, f'{v:.2f}s', va='center')

plt.tight_layout()
plt.savefig('results/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_comparison.png")

# 2. Confusion Matrices for all models
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')

for idx, (model_name, metrics) in enumerate(results.items()):
    ax = axes[idx // 4, idx % 4]
    cm = metrics['test']['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    ax.set_title(model_name)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('results/visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrices.png")

# 3. Metric comparison radar chart for top 3 models
comparison_df = pd.read_csv('results/model_comparison.csv')
top_3 = comparison_df.head(3)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
angles += angles[:1]

for idx, row in top_3.iterrows():
    model_name = row['Model']
    values = [results[model_name]['test'][m] for m in metrics_names]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
ax.set_ylim(0, 1)
ax.set_title('Top 3 Models Performance Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig('results/visualizations/top3_radar.png', dpi=300, bbox_inches='tight')
print("✅ Saved: top3_radar.png")

print("\n✅ All visualizations saved in results/visualizations/")
plt.show()