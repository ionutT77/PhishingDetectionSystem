"""
Data Splitting Script for Phishing Detection System
Splits the dataset into train/validation/test sets with stratified sampling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_data_splits(input_file, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets with stratified sampling
    
    Args:
        input_file: Path to the raw CSV file
        output_dir: Directory to save the split files
        train_ratio: Proportion of data for training (default: 0.8)
        val_ratio: Proportion of data for validation (default: 0.1)
        test_ratio: Proportion of data for testing (default: 0.1)
        random_state: Random seed for reproducibility
    """
    
    print("=" * 60)
    print("PHISHING DETECTION DATASET SPLITTING")
    print("=" * 60)
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} URLs")
    
    # Display dataset info
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total URLs: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Show class distribution
    print(f"\n📈 URL Type Distribution:")
    type_counts = df['type'].value_counts()
    for url_type, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {url_type:15s}: {count:7,} ({percentage:5.2f}%)")
    
    # First split: separate test set
    print(f"\n🔀 Splitting dataset...")
    print(f"   Train: {train_ratio*100:.0f}%")
    print(f"   Validation: {val_ratio*100:.0f}%")
    print(f"   Test: {test_ratio*100:.0f}%")
    
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        stratify=df['type'],
        random_state=random_state
    )
    
    # Second split: separate validation from train
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_df['type'],
        random_state=random_state
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'validation.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print(f"\n💾 Saving split datasets...")
    train_df.to_csv(train_path, index=False)
    print(f"   ✓ Train set saved: {train_path} ({len(train_df):,} URLs)")
    
    val_df.to_csv(val_path, index=False)
    print(f"   ✓ Validation set saved: {val_path} ({len(val_df):,} URLs)")
    
    test_df.to_csv(test_path, index=False)
    print(f"   ✓ Test set saved: {test_path} ({len(test_df):,} URLs)")
    
    # Verify stratification
    print(f"\n✅ Verifying stratification across splits:")
    print(f"\n{'Type':<15} {'Train':<12} {'Validation':<12} {'Test':<12}")
    print("-" * 60)
    
    for url_type in df['type'].unique():
        train_count = len(train_df[train_df['type'] == url_type])
        val_count = len(val_df[val_df['type'] == url_type])
        test_count = len(test_df[test_df['type'] == url_type])
        
        train_pct = (train_count / len(train_df)) * 100
        val_pct = (val_count / len(val_df)) * 100
        test_pct = (test_count / len(test_df)) * 100
        
        print(f"{url_type:<15} {train_count:6,} ({train_pct:4.1f}%)  {val_count:6,} ({val_pct:4.1f}%)  {test_count:6,} ({test_pct:4.1f}%)")
    
    print("\n" + "=" * 60)
    print("✅ Dataset splitting completed successfully!")
    print("=" * 60)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    INPUT_FILE = os.path.join(BASE_DIR, 'data', 'malicious_phish.csv')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    
    # Create splits
    train_df, val_df, test_df = create_data_splits(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )
