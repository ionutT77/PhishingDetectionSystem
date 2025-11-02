"""
Fix phishing_site_urls.csv column names
"""

import pandas as pd
from pathlib import Path

# Load the file
file_path = Path('data/raw/phishing/phishing_site_urls.csv')
df = pd.read_csv(file_path)

print(f"Original columns: {df.columns.tolist()}")
print(f"Original shape: {df.shape}")

# Rename columns to match expected format
df = df.rename(columns={'URL': 'url', 'Label': 'label'})

# Keep only phishing URLs (where label == 'bad')
df = df[df['label'] == 'bad'][['url']]

print(f"\nAfter processing:")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Save with correct format
df.to_csv(file_path, index=False)
print(f"\n✅ Fixed {file_path}")