"""
Feature Extraction for Large Dataset
"""

import pandas as pd
from pathlib import Path
import re
from urllib.parse import urlparse
import tldextract
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.features = []
    
    # ========== LEXICAL FEATURES =========
    
    def extract_url_length(self, url):
        """Total length of URL"""
        return len(url)
    
    def extract_domain_length(self, url):
        """Length of domain name"""
        try:
            domain = urlparse(url).netloc
            return len(domain)
        except:
            return 0
    
    def extract_path_length(self, url):
        """Length of URL path"""
        try:
            path = urlparse(url).path
            return len(path)
        except:
            return 0
    
    def has_ip_address(self, url):
        """Check if URL contains IP address instead of domain"""
        try:
            domain = urlparse(url).netloc
            ipv4_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            if re.search(ipv4_pattern, domain):
                return 1
            return 0
        except:
            return 0
    
    def count_dots(self, url):
        return url.count('.')
    
    def count_hyphens(self, url):
        return url.count('-')
    
    def count_underscores(self, url):
        return url.count('_')
    
    def count_slashes(self, url):
        return url.count('/')
    
    def count_question_marks(self, url):
        return url.count('?')
    
    def count_equals(self, url):
        return url.count('=')
    
    def count_at_symbol(self, url):
        return url.count('@')
    
    def count_ampersand(self, url):
        return url.count('&')
    
    def count_digits(self, url):
        return sum(c.isdigit() for c in url)
    
    def count_letters(self, url):
        return sum(c.isalpha() for c in url)
    
    def has_https(self, url):
        return 1 if url.startswith('https://') else 0
    
    def count_subdomains(self, url):
        try:
            extracted = tldextract.extract(url)
            subdomain = extracted.subdomain
            if subdomain:
                return len(subdomain.split('.'))
            return 0
        except:
            return 0
    
    def has_suspicious_words(self, url):
        suspicious_keywords = [
            'login', 'signin', 'bank', 'account', 'update', 'verify',
            'secure', 'ebay', 'paypal', 'amazon', 'facebook', 'apple',
            'microsoft', 'google', 'password', 'suspended', 'confirm'
        ]
        url_lower = url.lower()
        return sum(1 for keyword in suspicious_keywords if keyword in url_lower)
    
    def url_entropy(self, url):
        import math
        if not url:
            return 0
        
        entropy = 0
        for x in range(256):
            p_x = url.count(chr(x)) / len(url)
            if p_x > 0:
                entropy += - p_x * math.log2(p_x)
        return entropy
    
    def has_port(self, url):
        try:
            parsed = urlparse(url)
            return 1 if parsed.port else 0
        except:
            return 0
    
    def has_redirect(self, url):
        return 1 if '//' in url[8:] else 0
    
    def is_shortened_url(self, url):
        shortening_services = [
            'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly',
            'is.gd', 'buff.ly', 'adf.ly', 'short.to'
        ]
        domain = urlparse(url).netloc.lower()
        return 1 if any(service in domain for service in shortening_services) else 0
    
    def extract_features_from_url(self, url):
        """Extract all features from a single URL"""
        features = {
            'url_length': self.extract_url_length(url),
            'domain_length': self.extract_domain_length(url),
            'path_length': self.extract_path_length(url),
            'has_ip': self.has_ip_address(url),
            'count_dots': self.count_dots(url),
            'count_hyphens': self.count_hyphens(url),
            'count_underscores': self.count_underscores(url),
            'count_slashes': self.count_slashes(url),
            'count_question': self.count_question_marks(url),
            'count_equals': self.count_equals(url),
            'count_at': self.count_at_symbol(url),
            'count_ampersand': self.count_ampersand(url),
            'count_digits': self.count_digits(url),
            'count_letters': self.count_letters(url),
            'has_https': self.has_https(url),
            'count_subdomains': self.count_subdomains(url),
            'suspicious_words': self.has_suspicious_words(url),
            'url_entropy': self.url_entropy(url),
            'has_port': self.has_port(url),
            'has_redirect': self.has_redirect(url),
            'is_shortened': self.is_shortened_url(url),
        }
        return features
    
    def extract_features_from_dataframe(self, df, include_whois=False, sample_size=None):
        """Extract features from a DataFrame of URLs"""
        print("🔧 Extracting features from URLs...")
        print(f"   Total URLs: {len(df):,}")
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"   Sampling: {len(df):,} URLs for faster processing")
        
        feature_list = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"   Progress: {idx:,}/{len(df):,} URLs processed...")
            
            url = row['url']
            features = self.extract_features_from_url(url)
            
            if 'label' in row:
                features['label'] = row['label']
            
            feature_list.append(features)
        
        print(f"   ✅ Completed: {len(feature_list):,} URLs processed")
        
        return pd.DataFrame(feature_list)

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
        
        # Extract features
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