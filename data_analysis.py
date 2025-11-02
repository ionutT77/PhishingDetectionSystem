"""
Data Collection Module for Phishing Detection System
Collects URLs from multiple sources: PhishTank, OpenPhish, Tranco, etc.
"""
 # collecting data and find out how much data do i have
import pandas as pd
import requests
from datetime import datetime
import time
from pathlib import Path

class URLCollector:
    def __init__(self, output_dir='data/raw'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_openphish_urls(self, limit=10000):
        """
        Collect phishing URLs from OpenPhish
        Free feed: https://openphish.com/feed.txt
        """
        print("📥 Collecting URLs from OpenPhish...")
        try:
            url = "https://openphish.com/feed.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            urls = response.text.strip().split('\n')[:limit]
            
            df = pd.DataFrame({
                'url': urls,
                'label': 'phishing',
                'source': 'openphish',
                'verified': 'yes',
                'collected_date': datetime.now().isoformat()
            })
            
            output_path = self.output_dir / 'phishing' / 'openphish_urls.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Collected {len(df)} URLs from OpenPhish")
            return df
            
        except Exception as e:
            print(f"❌ Error collecting from OpenPhish: {e}")
            return pd.DataFrame()
    
    def collect_phishing_army_urls(self, limit=10000):
        """
        Collect phishing URLs from Phishing.Army
        Free blocklist: https://phishing.army/download/phishing_army_blocklist.txt
        """
        print("📥 Collecting URLs from Phishing.Army...")
        try:
            url = "https://phishing.army/download/phishing_army_blocklist.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the blocklist (skip comments)
            urls = []
            for line in response.text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Add http:// if not present
                    if not line.startswith('http'):
                        line = 'http://' + line
                    urls.append(line)
                if len(urls) >= limit:
                    break
            
            df = pd.DataFrame({
                'url': urls,
                'label': 'phishing',
                'source': 'phishing_army',
                'verified': 'yes',
                'collected_date': datetime.now().isoformat()
            })
            
            output_path = self.output_dir / 'phishing' / 'phishing_army_urls.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Collected {len(df)} URLs from Phishing.Army")
            return df
            
        except Exception as e:
            print(f"❌ Error collecting from Phishing.Army: {e}")
            return pd.DataFrame()
    
    def collect_certstream_suspicious_urls(self, limit=1000):
        """
        Collect suspicious domains from CertStream phishing catcher
        GitHub: https://raw.githubusercontent.com/x0rz/phishing_catcher/master/suspicious.txt
        """
        print("📥 Collecting URLs from CertStream Phishing Catcher...")
        try:
            url = "https://raw.githubusercontent.com/x0rz/phishing_catcher/master/suspicious.txt"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            domains = response.text.strip().split('\n')[:limit]
            urls = ['http://' + domain.strip() for domain in domains if domain.strip()]
            
            df = pd.DataFrame({
                'url': urls,
                'label': 'phishing',
                'source': 'certstream',
                'verified': 'suspected',
                'collected_date': datetime.now().isoformat()
            })
            
            output_path = self.output_dir / 'phishing' / 'certstream_urls.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Collected {len(df)} URLs from CertStream")
            return df
            
        except Exception as e:
            print(f"❌ Error collecting from CertStream: {e}")
            return pd.DataFrame()
    
    def collect_kaggle_phishing_dataset(self):
        """
        Download a public phishing dataset from a direct source
        Note: This uses a publicly available dataset mirror
        """
        print("📥 Collecting URLs from Kaggle Phishing Dataset...")
        try:
            # Using a public dataset repository
            url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset.csv"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Filter only phishing URLs
            if 'label' in df.columns or 'class' in df.columns:
                label_col = 'label' if 'label' in df.columns else 'class'
                phishing_df = df[df[label_col] == -1].copy()  # -1 typically means phishing
            else:
                phishing_df = df.copy()
            
            if 'url' in phishing_df.columns:
                phishing_df['source'] = 'kaggle'
                phishing_df['collected_date'] = datetime.now().isoformat()
                phishing_df['label'] = 'phishing'
                
                output_path = self.output_dir / 'phishing' / 'kaggle_urls.csv'
                output_path.parent.mkdir(parents=True, exist_ok=True)
                phishing_df.to_csv(output_path, index=False)
                print(f"✅ Collected {len(phishing_df)} URLs from Kaggle Dataset")
                return phishing_df
            else:
                print("⚠️  Kaggle dataset format unexpected, skipping...")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ Error collecting from Kaggle: {e}")
            return pd.DataFrame()
    
    def collect_tranco_urls(self, limit=10000):
        """
        Collect clean URLs from Tranco Top 1M list
        Download: https://tranco-list.eu/
        """
        print("📥 Collecting URLs from Tranco...")
        try:
            url = "https://tranco-list.eu/top-1m.csv.zip"
            response = requests.get(url, timeout=30)
            
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open('top-1m.csv') as f:
                    df = pd.read_csv(f, names=['rank', 'domain'], nrows=limit)
            
            df['url'] = 'http://' + df['domain']
            df['label'] = 'clean'
            df['source'] = 'tranco'
            df['collected_date'] = datetime.now().isoformat()
            
            output_path = self.output_dir / 'clean' / 'tranco_urls.csv'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Collected {len(df)} URLs from Tranco")
            return df
            
        except Exception as e:
            print(f"❌ Error collecting from Tranco: {e}")
            return pd.DataFrame()
    
    def collect_all(self):
        """Collect from all sources"""
        print("\n🚀 Starting data collection...")
        print("=" * 50)
        
        # Collect phishing URLs from multiple sources
        phishing_dfs = []
        
        openphish_df = self.collect_openphish_urls()
        if not openphish_df.empty:
            phishing_dfs.append(openphish_df)
        time.sleep(2)
        
        phishing_army_df = self.collect_phishing_army_urls()
        if not phishing_army_df.empty:
            phishing_dfs.append(phishing_army_df)
        time.sleep(2)
        
        certstream_df = self.collect_certstream_suspicious_urls()
        if not certstream_df.empty:
            phishing_dfs.append(certstream_df)
        time.sleep(2)
        
        kaggle_df = self.collect_kaggle_phishing_dataset()
        if not kaggle_df.empty:
            phishing_dfs.append(kaggle_df)
        time.sleep(2)
        
        # Collect clean URLs
        clean_df = self.collect_tranco_urls()
        
        # Combine all phishing sources
        if phishing_dfs:
            all_phishing = pd.concat(phishing_dfs, ignore_index=True)
            # Remove duplicates
            all_phishing = all_phishing.drop_duplicates(subset=['url'])
            
            # Save combined phishing dataset
            output_path = self.output_dir / 'phishing' / 'all_phishing_urls.csv'
            all_phishing.to_csv(output_path, index=False)
            
            print("\n📊 Phishing URLs by Source:")
            if 'source' in all_phishing.columns:
                print(all_phishing['source'].value_counts().to_string())
        else:
            all_phishing = pd.DataFrame()
        
        print("\n📊 Collection Summary:")
        print(f"  Phishing URLs: {len(all_phishing)}")
        print(f"  Clean URLs: {len(clean_df)}")
        print(f"  Total: {len(all_phishing) + len(clean_df)}")
        print(f"\n✅ Data saved to: {self.output_dir}")
        
        return all_phishing, clean_df

if __name__ == "__main__":
    collector = URLCollector()
    collector.collect_all()

"""
Data Analysis Script - Check collected data
"""

import pandas as pd
from pathlib import Path

def analyze_collected_data():
    """Analyze all collected data"""
    print("🔍 Analyzing Collected Data")
    print("=" * 60)
    
    data_dir = Path('data/raw')
    
    # Check phishing data
    phishing_dir = data_dir / 'phishing'
    clean_dir = data_dir / 'clean'
    
    total_phishing = 0
    total_clean = 0
    
    print("\n📊 PHISHING DATA:")
    print("-" * 60)
    
    if phishing_dir.exists():
        for csv_file in phishing_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                print(f"  {csv_file.name:30s} : {len(df):,} URLs")
                total_phishing += len(df)
            except Exception as e:
                print(f"  ❌ Error reading {csv_file.name}: {e}")
    else:
        print("  ⚠️  No phishing directory found")
    
    print("\n📊 CLEAN DATA:")
    print("-" * 60)
    
    if clean_dir.exists():
        for csv_file in clean_dir.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file)
                print(f"  {csv_file.name:30s} : {len(df):,} URLs")
                total_clean += len(df)
            except Exception as e:
                print(f"  ❌ Error reading {csv_file.name}: {e}")
    else:
        print("  ⚠️  No clean directory found")
    
    print("\n" + "=" * 60)
    print(f"📈 TOTAL PHISHING URLs: {total_phishing:,}")
    print(f"📈 TOTAL CLEAN URLs: {total_clean:,}")
    print(f"📈 TOTAL DATASET SIZE: {total_phishing + total_clean:,}")
    
    # Check balance
    if total_phishing > 0 and total_clean > 0:
        ratio = total_phishing / total_clean
        print(f"\n⚖️  Dataset Balance Ratio: {ratio:.2f} (phishing/clean)")
        
        if 0.3 <= ratio <= 3.0:
            print("✅ Good balance! Dataset is suitable for training.")
        elif ratio < 0.3:
            print("⚠️  Low phishing ratio. Consider collecting more phishing URLs.")
        else:
            print("⚠️  High phishing ratio. Consider balancing the dataset.")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    
    if total_phishing >= 5000 and total_clean >= 5000:
        print("✅ Excellent! You have enough data for robust training.")
        print("   Recommended split: 70% train, 15% validation, 15% test")
    elif total_phishing >= 2000 and total_clean >= 2000:
        print("✅ Good! You have sufficient data for training.")
        print("   Consider collecting more for better model performance.")
    elif total_phishing >= 1000 and total_clean >= 1000:
        print("⚠️  Minimum viable dataset. Training is possible but:")
        print("   - Use cross-validation")
        print("   - Consider data augmentation")
        print("   - Collect more data if possible")
    else:
        print("❌ Insufficient data. Recommendations:")
        print("   - Collect at least 1,000 URLs per class")
        print("   - Download additional Kaggle datasets")
        print("   - Use PhishTank API for more data")
    
    # Check for combined dataset
    combined_phishing = phishing_dir / 'all_phishing_urls.csv'
    if combined_phishing.exists():
        print("\n📦 Combined Dataset:")
        df = pd.read_csv(combined_phishing)
        print(f"   all_phishing_urls.csv: {len(df):,} unique URLs")
        
        if 'source' in df.columns:
            print("\n   Sources breakdown:")
            for source, count in df['source'].value_counts().items():
                print(f"     - {source}: {count:,}")

if __name__ == "__main__":
    analyze_collected_data()