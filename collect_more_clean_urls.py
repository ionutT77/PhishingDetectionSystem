"""
Collect additional clean URLs to balance large dataset
"""

import pandas as pd
import requests
from pathlib import Path
import time

def collect_additional_clean_urls():
    """Collect clean URLs from multiple sources"""
    print("🌐 Collecting Additional Clean URLs")
    print("=" * 60)
    
    output_dir = Path('data/raw/clean')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_clean = []
    
    # 1. Cisco Umbrella Top 1M
    print("\n📥 Collecting from Cisco Umbrella...")
    try:
        url = "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
        response = requests.get(url, timeout=60)
        
        import zipfile, io
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open('top-1m.csv') as f:
                df = pd.read_csv(f, names=['rank', 'domain'], nrows=50000)
        
        df['url'] = 'https://' + df['domain']
        df['source'] = 'umbrella'
        all_clean.append(df)
        print(f"   ✅ Collected {len(df):,} URLs")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    time.sleep(2)
    
    # 2. Majestic Million
    print("\n📥 Collecting from Majestic Million...")
    try:
        url = "https://downloads.majestic.com/majestic_million.csv"
        df = pd.read_csv(url, nrows=50000)
        
        if 'Domain' in df.columns:
            df['url'] = 'https://' + df['Domain']
            df['source'] = 'majestic'
            all_clean.append(df[['url', 'source']])
            print(f"   ✅ Collected {len(df):,} URLs")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 3. Combine all
    if all_clean:
        combined = pd.concat(all_clean, ignore_index=True)
        combined = combined[['url', 'source']].drop_duplicates(subset=['url'])
        combined['label'] = 0
        
        output_path = output_dir / 'additional_clean_urls.csv'
        combined.to_csv(output_path, index=False)
        
        print(f"\n✅ Total collected: {len(combined):,} clean URLs")
        print(f"📁 Saved to: {output_path}")
        
        # Now merge with existing
        existing_path = output_dir / 'tranco_urls.csv'
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            total = pd.concat([existing, combined], ignore_index=True)
            total = total.drop_duplicates(subset=['url'])
            
            merged_path = output_dir / 'all_clean_urls.csv'
            total.to_csv(merged_path, index=False)
            print(f"\n📦 Merged with existing: {len(total):,} total clean URLs")
            print(f"📁 Saved to: {merged_path}")

if __name__ == "__main__":
    collect_additional_clean_urls()