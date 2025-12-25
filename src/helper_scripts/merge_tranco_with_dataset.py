"""
Merge Tranco benign URLs with existing dataset
Checks for duplicates and creates balanced final dataset
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

print("="*70)
print("MERGE TRANCO URLs WITH EXISTING DATASET (WITH DUPLICATE CHECK)")
print("="*70)

# Paths
EXISTING_DIR = Path('../../data/processed/2nd_try_1mil_566k')
GENERATED_SYNTHETIC = Path('../../data/raw/benign_urls_generated_300k.csv')
TRANCO_FILE = Path('../../data/raw/tranco_benign_urls.csv')
OUTPUT_DIR = Path('../../data/processed/final_balanced')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load existing dataset
print("\n📥 Loading existing dataset...")
existing_train = pd.read_csv(EXISTING_DIR / 'train.csv')
existing_val = pd.read_csv(EXISTING_DIR / 'validation.csv')
existing_test = pd.read_csv(EXISTING_DIR / 'test.csv')

# Combine to check for duplicates
existing_combined = pd.concat([existing_train, existing_val, existing_test], ignore_index=True)
print(f"   Existing total: {len(existing_combined):,} URLs")

# Count existing by label
print(f"\n📊 Existing distribution:")
for label in ['benign', 'defacement', 'malware', 'phishing']:
    count = (existing_combined['label'] == label).sum()
    pct = count / len(existing_combined) * 100
    print(f"   {label:12}: {count:7,} ({pct:5.2f}%)")

# Load new benign URLs
new_benign_dfs = []

# Load synthetic generated URLs (if exists)
if GENERATED_SYNTHETIC.exists():
    print(f"\n📥 Loading synthetic generated URLs...")
    synthetic_df = pd.read_csv(GENERATED_SYNTHETIC)
    print(f"   Synthetic: {len(synthetic_df):,}")
    new_benign_dfs.append(synthetic_df)
else:
    print(f"\n⚠️ Synthetic URLs file not found: {GENERATED_SYNTHETIC}")

# Load Tranco URLs
if TRANCO_FILE.exists():
    print(f"\n📥 Loading Tranco URLs...")
    tranco_df = pd.read_csv(TRANCO_FILE)
    print(f"   Tranco: {len(tranco_df):,}")
    new_benign_dfs.append(tranco_df)
else:
    print(f"\n❌ Tranco URLs file not found: {TRANCO_FILE}")
    print(f"   Run download_tranco_urls.py first!")
    exit(1)

# Combine all new benign URLs
new_benign = pd.concat(new_benign_dfs, ignore_index=True)
print(f"\n📊 Total new benign URLs: {len(new_benign):,}")

# CRITICAL: Remove duplicates within new benign URLs
print(f"\n🔍 Removing duplicates within new benign URLs...")
initial_new = len(new_benign)
new_benign = new_benign.drop_duplicates(subset=['url'], keep='first')
new_duplicates = initial_new - len(new_benign)
print(f"   Initial: {initial_new:,}")
print(f"   After deduplication: {len(new_benign):,}")
print(f"   Removed: {new_duplicates:,} duplicates")

# Check for duplicates with existing dataset
print(f"\n🔍 Checking for duplicates with existing dataset...")
existing_urls = set(existing_combined['url'].str.lower())
new_benign['url_lower'] = new_benign['url'].str.lower()

# Mark duplicates
new_benign['is_duplicate'] = new_benign['url_lower'].isin(existing_urls)
duplicates_with_existing = new_benign['is_duplicate'].sum()

print(f"   Duplicates with existing dataset: {duplicates_with_existing:,}")

# Remove duplicates
new_benign = new_benign[~new_benign['is_duplicate']].copy()
new_benign = new_benign[['url', 'label']]  # Keep only needed columns

print(f"   New benign URLs after duplicate removal: {len(new_benign):,}")

# Split new benign URLs into train/val/test (70/15/15)
print(f"\n🔀 Splitting new benign URLs into train/val/test...")
train_new, temp = train_test_split(new_benign, test_size=0.3, random_state=42)
val_new, test_new = train_test_split(temp, test_size=0.5, random_state=42)

print(f"   Train: {len(train_new):,}")
print(f"   Val:   {len(val_new):,}")
print(f"   Test:  {len(test_new):,}")

# Merge with existing datasets
print(f"\n🔗 Merging with existing datasets...")
final_train = pd.concat([existing_train, train_new], ignore_index=True)
final_val = pd.concat([existing_val, val_new], ignore_index=True)
final_test = pd.concat([existing_test, test_new], ignore_index=True)

# Shuffle
final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)
final_val = final_val.sample(frac=1, random_state=42).reset_index(drop=True)
final_test = final_test.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"   Final Train: {len(final_train):,}")
print(f"   Final Val:   {len(final_val):,}")
print(f"   Final Test:  {len(final_test):,}")

# Show final distribution
print(f"\n📊 Final distribution:")
for df_name, df in [('Train', final_train), ('Val', final_val), ('Test', final_test)]:
    print(f"\n{df_name}: {len(df):,} total")
    for label in ['benign', 'defacement', 'malware', 'phishing']:
        count = (df['label'] == label).sum()
        pct = count / len(df) * 100
        print(f"   {label:12}: {count:7,} ({pct:5.2f}%)")

# Calculate imbalance ratio
train_counts = final_train['label'].value_counts()
max_count = train_counts.max()
min_count = train_counts.min()
imbalance_ratio = max_count / min_count

benign_count = (final_train['label'] == 'benign').sum()
phishing_count = (final_train['label'] == 'phishing').sum()

print(f"\n⚖️ Balance Analysis:")
print(f"   Benign:   {benign_count:,}")
print(f"   Phishing: {phishing_count:,}")
print(f"   Ratio: {phishing_count/benign_count:.2f}:1 (phishing:benign)")
print(f"   Overall imbalance: {imbalance_ratio:.2f}:1 (max:min)")

if benign_count > phishing_count:
    print("   ✅ Benign is now the majority class!")
elif phishing_count / benign_count < 1.5:
    print("   ✅ Well balanced!")
else:
    print("   ⚠️ Still slightly phishing-heavy")

# Save final datasets
print(f"\n💾 Saving final datasets...")
final_train.to_csv(OUTPUT_DIR / 'train.csv', index=False)
final_val.to_csv(OUTPUT_DIR / 'validation.csv', index=False)
final_test.to_csv(OUTPUT_DIR / 'test.csv', index=False)

print(f"   ✅ Saved: {OUTPUT_DIR / 'train.csv'}")
print(f"   ✅ Saved: {OUTPUT_DIR / 'validation.csv'}")
print(f"   ✅ Saved: {OUTPUT_DIR / 'test.csv'}")

# Summary
total_original = len(existing_combined)
total_final = len(final_train) + len(final_val) + len(final_test)
increase = total_final - total_original

print(f"\n" + "="*70)
print("✅ MERGE COMPLETE!")
print("="*70)
print(f"   Original dataset:    {total_original:,} URLs")
print(f"   New benign added:    {len(new_benign):,} URLs")
print(f"   Duplicates removed:  {new_duplicates + duplicates_with_existing:,} URLs")
print(f"   Final dataset:       {total_final:,} URLs")
print(f"   Net increase:        +{increase:,} URLs ({increase/total_original*100:.1f}%)")

print(f"\n📁 Final dataset location: {OUTPUT_DIR}")

print(f"\n🚀 Next steps:")
print(f"   1. Upload CSVs from '{OUTPUT_DIR.name}' to Kaggle")
print(f"   2. Create/update Kaggle dataset")
print(f"   3. Update notebook path to use new dataset")
print(f"   4. Retrain model (expected time: ~35-40 mins)")
print(f"\n💡 Expected improvement:")
print(f"   With {benign_count:,} real benign URLs from Tranco,")
print(f"   famous brands should now predict correctly as benign!")
print("="*70)
