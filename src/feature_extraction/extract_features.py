"""
Feature Extraction for Large Dataset
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.append('src/feature_extraction')
from extract_features import URLFeatureExtractor

def main():
    print("🚀 Feature Extraction for Large Dataset")
    print("="*60)
    
    data_dir = Path('data/processed')
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = URLFeatureExtractor()
    
    # Process large datasets
    datasets = ['train_large.csv', 'val_large.csv', 'test_large.csv']
    
    for dataset_name in datasets:
        dataset_path = data_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"⚠️  {dataset_name} not found, skipping...")
            continue
        
        print(f"\n📂 Processing {dataset_name}...")
        df = pd.read_csv(dataset_path)
        
        # Extract features (without WHOIS for speed)
        features_df = extractor.extract_features_from_dataframe(
            df,
            include_whois=False
        )
        
        # Save features
        output_name = f'features_{dataset_name}'
        output_path = output_dir / output_name
        features_df.to_csv(output_path, index=False)
        print(f"   ✅ Saved to: {output_path}")
        print(f"   Shape: {features_df.shape}")
    
    print("\n" + "="*60)
    print("✅ Feature extraction completed!")
    print(f"📁 Features saved to: {output_dir}/")
    print("\n🎯 NEXT STEP: Train models")
    print("   python src/models/train_models.py")
    print("="*60)

if __name__ == "__main__":
    main()