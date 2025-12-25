"""
Check Label Distribution in Training Dataset
Analyzes the percentage of each class in train, validation, and test sets
"""

import pandas as pd
from pathlib import Path

# Paths to dataset files
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "2nd_try_1mil_566k"

train_file = DATA_DIR / "train.csv"
val_file = DATA_DIR / "validation.csv"
test_file = DATA_DIR / "test.csv"

print("=" * 70)
print("📊 LABEL DISTRIBUTION ANALYSIS")
print("=" * 70)

# Load datasets
print("\n📥 Loading datasets...")
train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

print(f"✅ Train: {len(train_df):,} URLs")
print(f"✅ Val:   {len(val_df):,} URLs")
print(f"✅ Test:  {len(test_df):,} URLs")
print(f"✅ Total: {len(train_df) + len(val_df) + len(test_df):,} URLs")

# Analyze each dataset
datasets = {
    'Training': train_df,
    'Validation': val_df,
    'Test': test_df
}

all_labels = set()
for df in datasets.values():
    all_labels.update(df['label'].unique())

labels = sorted(list(all_labels))

print("\n" + "=" * 70)
print("📊 LABEL DISTRIBUTION BY DATASET")
print("=" * 70)

for dataset_name, df in datasets.items():
    print(f"\n{dataset_name} Set ({len(df):,} URLs):")
    print("-" * 50)
    
    label_counts = df['label'].value_counts()
    
    for label in labels:
        count = label_counts.get(label, 0)
        percentage = (count / len(df)) * 100
        print(f"  {label:15} {count:8,} URLs  ({percentage:5.2f}%)")

# Overall distribution
print("\n" + "=" * 70)
print("📊 OVERALL DISTRIBUTION (All Sets Combined)")
print("=" * 70)

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
total_urls = len(all_df)

print(f"\nTotal URLs: {total_urls:,}\n")
print("-" * 50)

label_counts = all_df['label'].value_counts()

for label in labels:
    count = label_counts.get(label, 0)
    percentage = (count / total_urls) * 100
    print(f"  {label:15} {count:8,} URLs  ({percentage:5.2f}%)")

# Check for imbalance
print("\n" + "=" * 70)
print("⚖️ IMBALANCE ANALYSIS")
print("=" * 70)

max_count = label_counts.max()
min_count = label_counts.min()
imbalance_ratio = max_count / min_count

print(f"\nMost common class:  {label_counts.idxmax()} ({max_count:,} URLs)")
print(f"Least common class: {label_counts.idxmin()} ({min_count:,} URLs)")
print(f"Imbalance ratio:    {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("\n⚠️ WARNING: Significant class imbalance detected!")
    print("   Recommendation: Use class_weight='balanced' during training")
    print("   This will help the model learn all classes equally well.")
elif imbalance_ratio > 2:
    print("\n⚠️ Moderate class imbalance detected")
    print("   Consider using class weights during training")
else:
    print("\n✅ Dataset is reasonably balanced")

print("\n" + "=" * 70)
