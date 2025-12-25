"""
Create character-to-index mapping for the neural network model
This matches the preprocessing used during training
"""

import pickle
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "final_balanced_2.238mil" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "results_2mil238k_dataset" / "char_to_idx.pkl"

print("=" * 60)
print("🔧 Creating Character Mapping for Neural Network")
print("=" * 60)

# Load training data
print(f"\n📂 Loading training data from: {TRAIN_DATA_PATH}")
try:
    # Load a sample to extract characters (don't need all rows)
    df = pd.read_csv(TRAIN_DATA_PATH, nrows=100000)
    print(f"✅ Loaded {len(df):,} URLs (sample)")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# Get URL column
url_column = 'url' if 'url' in df.columns else df.columns[0]
print(f"🔗 Using column: '{url_column}'")

# Extract all unique characters
print("\n🔍 Extracting unique characters from URLs...")
all_chars = set()
for url in df[url_column]:
    all_chars.update(url)

chars = sorted(list(all_chars))
print(f"✅ Found {len(chars)} unique characters")
print(f"   Sample: {chars[:20]}")

# Create mapping (0 reserved for padding)
char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}
idx_to_char = {i+1: ch for i, ch in enumerate(chars)}
vocab_size = len(chars) + 1

print(f"\n📊 Vocabulary size: {vocab_size}")
print(f"   (includes 0 for padding)")

# Save the mapping
print(f"\n💾 Saving character mapping to: {OUTPUT_PATH}")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

mapping = {
    'char_to_idx': char_to_idx,
    'idx_to_char': idx_to_char,
    'vocab_size': vocab_size,
    'max_url_len': 200  # Same as training
}

try:
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(mapping, f)
    print(f"✅ Character mapping saved!")
    
    # Verify file size
    file_size = OUTPUT_PATH.stat().st_size / 1024  # KB
    print(f"   File size: {file_size:.2f} KB")
except Exception as e:
    print(f"❌ Error saving mapping: {e}")
    exit(1)

# Test the mapping
print("\n🧪 Testing character mapping...")
test_url = "https://www.google.com"
encoded = [char_to_idx.get(c, 0) for c in test_url[:200]]
print(f"   Test URL: {test_url}")
print(f"   Encoded length: {len(encoded)}")
print(f"   First 10 indices: {encoded[:10]}")

print("\n" + "=" * 60)
print("✅ ALL DONE!")
print("=" * 60)
print(f"\n📌 Character mapping saved to: {OUTPUT_PATH}")
print("🚀 Now update the UI to use character encoding")
print("=" * 60)
