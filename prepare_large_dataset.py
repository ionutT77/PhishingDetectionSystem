"""
Prepare Larger Balanced Dataset for Training
Uses more data for better model performance
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_large_dataset(phishing_size=50000, clean_size=50000, random_state=42):
    """
    Create a larger balanced dataset (100K total)
    
    Args:
        phishing_size: Number of phishing URLs (default: 50K)
        clean_size: Number of clean URLs (default: 50K)
    """
    print("🔄 Preparing LARGE Balanced Dataset...")
    print("=" * 60)
    
    data_dir = Path('data/raw')
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ALL phishing URLs from all files
    phishing_dir = data_dir / 'phishing'
    print(f"📥 Loading ALL phishing URLs from {phishing_dir}...")
    
    phishing_dfs = []
    for csv_file in phishing_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            
            # Handle different column naming conventions
            if 'URL' in df.columns:  # Capital URL
                df = df.rename(columns={'URL': 'url'})
            
            if 'url' in df.columns:
                # Only keep the url column
                phishing_dfs.append(df[['url']])
                print(f"   ✓ {csv_file.name}: {len(df):,} URLs")
            else:
                print(f"   ⚠️  {csv_file.name}: No 'url' or 'URL' column found")
        except Exception as e:
            print(f"   ❌ Error reading {csv_file.name}: {e}")
    
    if phishing_dfs:
        phishing_df = pd.concat(phishing_dfs, ignore_index=True)
        phishing_df = phishing_df.drop_duplicates(subset=['url'])
        print(f"   📊 Total unique phishing URLs: {len(phishing_df):,}")
    else:
        print("   ❌ No phishing URLs found!")
        return None
    
    # Load ALL clean URLs from all files
    clean_dir = data_dir / 'clean'
    print(f"\n📥 Loading ALL clean URLs from {clean_dir}...")
    
    clean_dfs = []
    for csv_file in clean_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            if 'url' in df.columns:
                clean_dfs.append(df[['url']])
                print(f"   ✓ {csv_file.name}: {len(df):,} URLs")
        except Exception as e:
            print(f"   ❌ Error reading {csv_file.name}: {e}")
    
    if clean_dfs:
        clean_df = pd.concat(clean_dfs, ignore_index=True)
        clean_df = clean_df.drop_duplicates(subset=['url'])
        print(f"   📊 Total unique clean URLs: {len(clean_df):,}")
    else:
        print("   ❌ No clean URLs found!")
        return None
    
    # Sample requested amounts
    print(f"\n✂️  Sampling datasets...")
    
    if len(phishing_df) > phishing_size:
        phishing_df = phishing_df.sample(n=phishing_size, random_state=random_state)
        print(f"   Phishing: sampled {len(phishing_df):,} from {len(phishing_df):,} available")
    else:
        print(f"   Phishing: using all {len(phishing_df):,} URLs")
    
    if len(clean_df) > clean_size:
        clean_df = clean_df.sample(n=clean_size, random_state=random_state)
        print(f"   Clean: sampled {len(clean_df):,} from {len(clean_df):,} available")
    else:
        print(f"   Clean: using all {len(clean_df):,} URLs")
    
    # Balance the dataset
    final_size = min(len(phishing_df), len(clean_df))
    if len(phishing_df) != len(clean_df):
        print(f"\n⚖️  Balancing dataset to {final_size:,} samples per class...")
        phishing_df = phishing_df.sample(n=final_size, random_state=random_state)
        clean_df = clean_df.sample(n=final_size, random_state=random_state)
    
    # Prepare datasets
    phishing_df = phishing_df[['url']].copy()
    clean_df = clean_df[['url']].copy()
    
    phishing_df['label'] = 1
    clean_df['label'] = 0
    
    combined_df = pd.concat([phishing_df, clean_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n📊 Final Dataset Summary:")
    print(f"   Total: {len(combined_df):,} URLs")
    print(f"   Phishing (label=1): {(combined_df['label']==1).sum():,} ({(combined_df['label']==1).sum()/len(combined_df)*100:.1f}%)")
    print(f"   Clean (label=0): {(combined_df['label']==0).sum():,} ({(combined_df['label']==0).sum()/len(combined_df)*100:.1f}%)")
    
    # Split: 70% train, 15% val, 15% test
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
    train_df.to_csv(output_dir / 'train_large.csv', index=False)
    val_df.to_csv(output_dir / 'val_large.csv', index=False)
    test_df.to_csv(output_dir / 'test_large.csv', index=False)
    combined_df.to_csv(output_dir / 'combined_large.csv', index=False)
    
    print(f"\n✅ Datasets saved to: {output_dir}/")
    print(f"   - train_large.csv ({len(train_df):,} URLs)")
    print(f"   - val_large.csv ({len(val_df):,} URLs)")
    print(f"   - test_large.csv ({len(test_df):,} URLs)")
    print(f"   - combined_large.csv ({len(combined_df):,} URLs)")
    
    # Estimate times
    total_urls = len(combined_df)
    est_feature_time = (total_urls / 1000) * 0.5  # ~0.5 min per 1K URLs
    est_train_time = est_feature_time / 5
    
    print(f"\n⏱️  Estimated Times:")
    print(f"   Feature extraction: ~{est_feature_time:.0f} minutes")
    print(f"   Model training: ~{est_train_time:.0f} minutes")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    print("🚀 Creating Large Balanced Dataset\n")
    
    # Choose your dataset size:
    
    # Option 1: 100K total (50K per class) - Good balance of speed and performance
    result = prepare_large_dataset(phishing_size=50000, clean_size=50000)
    
    # Option 2: 200K total (100K per class) - Better performance, longer training
    # result = prepare_large_dataset(phishing_size=100000, clean_size=100000)
    
    # Option 3: Use maximum available data
    # result = prepare_large_dataset(phishing_size=500000, clean_size=100000)
    
    if result:
        print("\n" + "="*60)
        print("🎯 NEXT STEPS:")
        print("="*60)
        print("1. Extract features: python extract_features_large.py")
        print("2. Train models: python src/models/train_models.py")
        print("="*60)