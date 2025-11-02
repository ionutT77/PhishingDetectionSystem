"""
Prepare Balanced Dataset for Training
Combines and balances phishing and clean URLs
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_balanced_dataset(phishing_sample_size=10000, clean_sample_size=10000, random_state=42):
    """
    Create a balanced dataset for training
    
    Args:
        phishing_sample_size: Number of phishing URLs to sample
        clean_sample_size: Number of clean URLs to sample
        random_state: Random seed for reproducibility
    """
    print("🔄 Preparing Balanced Dataset...")
    print("=" * 60)
    
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all phishing URLs
    phishing_path = data_dir / 'phishing' / 'all_phishing_urls.csv'
    if phishing_path.exists():
        print(f"📥 Loading phishing URLs from {phishing_path}")
        phishing_df = pd.read_csv(phishing_path)
        print(f"   Total available: {len(phishing_df):,}")
        
        # Sample phishing URLs
        if len(phishing_df) > phishing_sample_size:
            phishing_df = phishing_df.sample(n=phishing_sample_size, random_state=random_state)
            print(f"   ✂️  Sampled to: {len(phishing_df):,}")
    else:
        print("❌ No combined phishing file found!")
        print(f"   Looking for: {phishing_path}")
        return None
    
    # Load all clean URLs
    clean_path = data_dir / 'clean' / 'tranco_urls.csv'
    if clean_path.exists():
        print(f"\n📥 Loading clean URLs from {clean_path}")
        clean_df = pd.read_csv(clean_path)
        print(f"   Total available: {len(clean_df):,}")
        
        # Sample clean URLs
        if len(clean_df) > clean_sample_size:
            clean_df = clean_df.sample(n=clean_sample_size, random_state=random_state)
            print(f"   ✂️  Sampled to: {len(clean_df):,}")
    else:
        print("❌ No clean URLs file found!")
        print(f"   Looking for: {clean_path}")
        return None
    
    # Keep only necessary columns
    phishing_df = phishing_df[['url']].copy()
    clean_df = clean_df[['url']].copy()
    
    # Create binary labels (1 = phishing, 0 = clean)
    phishing_df['label'] = 1
    clean_df['label'] = 0
    
    # Combine datasets
    combined_df = pd.concat([phishing_df, clean_df], ignore_index=True)
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n📊 Combined Dataset Summary:")
    print(f"   Total URLs: {len(combined_df):,}")
    print(f"   Phishing (label=1): {(combined_df['label']==1).sum():,} ({(combined_df['label']==1).sum()/len(combined_df)*100:.1f}%)")
    print(f"   Clean (label=0): {(combined_df['label']==0).sum():,} ({(combined_df['label']==0).sum()/len(combined_df)*100:.1f}%)")
    
    # Split into train, validation, test sets
    # 70% train, 15% validation, 15% test
    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=random_state, stratify=combined_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=random_state, stratify=temp_df['label'])
    
    print(f"\n📂 Dataset Splits:")
    print(f"   Training:   {len(train_df):,} URLs ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"     - Phishing: {(train_df['label']==1).sum():,}")
    print(f"     - Clean: {(train_df['label']==0).sum():,}")
    print(f"   Validation: {len(val_df):,} URLs ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"     - Phishing: {(val_df['label']==1).sum():,}")
    print(f"     - Clean: {(val_df['label']==0).sum():,}")
    print(f"   Test:       {len(test_df):,} URLs ({len(test_df)/len(combined_df)*100:.1f}%)")
    print(f"     - Phishing: {(test_df['label']==1).sum():,}")
    print(f"     - Clean: {(test_df['label']==0).sum():,}")
    
    # Save datasets
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    combined_df.to_csv(output_dir / 'combined_balanced.csv', index=False)
    
    print(f"\n✅ Datasets saved to: {output_dir}/")
    print(f"   - train.csv ({len(train_df):,} URLs)")
    print(f"   - val.csv ({len(val_df):,} URLs)")
    print(f"   - test.csv ({len(test_df):,} URLs)")
    print(f"   - combined_balanced.csv ({len(combined_df):,} URLs)")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Create balanced dataset with 10K URLs per class (20K total)
    print("🚀 Creating Balanced Dataset for Training\n")
    
    result = prepare_balanced_dataset(
        phishing_sample_size=10000,
        clean_sample_size=10000,
        random_state=42
    )
    
    if result:
        print("\n" + "="*60)
        print("🎯 NEXT STEPS:")
        print("="*60)
        print("✅ Week 1-2: Data Collection - COMPLETED!")
        print("➡️  Week 3: Feature Extraction - RUN THIS NEXT:")
        print("   python src/feature_extraction/extract_features.py")
        print("="*60)
    else:
        print("\n❌ Failed to prepare dataset. Check the error messages above.")