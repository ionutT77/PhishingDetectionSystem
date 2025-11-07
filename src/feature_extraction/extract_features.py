"""
Feature Extraction Module for Phishing Detection
Extracts various features from URLs for ML model training
"""
# this is a script for feature extraction like

# Lexical Features (21):

# URL length, domain length, path length
# Special character counts (dots, hyphens, @, etc.)
# Has HTTPS, has IP address
# Number of subdomains
# Suspicious keywords count
# URL entropy (randomness)
# URL shortener detection
# Optional Host Features (slower):

# Domain age (WHOIS) - disabled by default
# DNS record check

import pandas as pd
import re
from urllib.parse import urlparse
import tldextract
from pathlib import Path
import socket
import whois
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class URLFeatureExtractor:
    """Extract features from URLs for phishing detection"""
    
    def __init__(self):
        self.features = []
    
    # ========== LEXICAL FEATURES ==========
    
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
            # Check for IPv4 pattern
            ipv4_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            if re.search(ipv4_pattern, domain):
                return 1
            return 0
        except:
            return 0
    
    def count_dots(self, url):
        """Count number of dots in URL"""
        return url.count('.')
    
    def count_hyphens(self, url):
        """Count number of hyphens in URL"""
        return url.count('-')
    
    def count_underscores(self, url):
        """Count number of underscores in URL"""
        return url.count('_')
    
    def count_slashes(self, url):
        """Count number of slashes in URL"""
        return url.count('/')
    
    def count_question_marks(self, url):
        """Count number of question marks in URL"""
        return url.count('?')
    
    def count_equals(self, url):
        """Count number of equals signs in URL"""
        return url.count('=')
    
    def count_at_symbol(self, url):
        """Count number of @ symbols in URL"""
        return url.count('@')
    
    def count_ampersand(self, url):
        """Count number of & symbols in URL"""
        return url.count('&')
    
    def count_digits(self, url):
        """Count number of digits in URL"""
        return sum(c.isdigit() for c in url)
    
    def count_letters(self, url):
        """Count number of letters in URL"""
        return sum(c.isalpha() for c in url)
    
    def has_https(self, url):
        """Check if URL uses HTTPS"""
        return 1 if url.startswith('https://') else 0
    
    def count_subdomains(self, url):
        """Count number of subdomains"""
        try:
            extracted = tldextract.extract(url)
            subdomain = extracted.subdomain
            if subdomain:
                return len(subdomain.split('.'))
            return 0
        except:
            return 0
    
    def has_suspicious_words(self, url):
        """Check for suspicious keywords in URL"""
        suspicious_keywords = [
            'login', 'signin', 'bank', 'account', 'update', 'verify',
            'secure', 'ebay', 'paypal', 'amazon', 'facebook', 'apple',
            'microsoft', 'google', 'password', 'suspended', 'confirm'
        ]
        url_lower = url.lower()
        return sum(1 for keyword in suspicious_keywords if keyword in url_lower)
    
    def url_entropy(self, url):
        """Calculate Shannon entropy of URL"""
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
        """Check if URL specifies a port"""
        try:
            parsed = urlparse(url)
            return 1 if parsed.port else 0
        except:
            return 0
    
    def has_redirect(self, url):
        """Check for redirect patterns (//)"""
        return 1 if '//' in url[8:] else 0  # Skip protocol
    
    def is_shortened_url(self, url):
        """Check if URL is from a URL shortening service"""
        shortening_services = [
            'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'ow.ly',
            'is.gd', 'buff.ly', 'adf.ly', 'short.to'
        ]
        domain = urlparse(url).netloc.lower()
        return 1 if any(service in domain for service in shortening_services) else 0
    
    # ========== HOST-BASED FEATURES ==========
    
    def get_domain_age(self, url):
        """Get domain age in days using WHOIS (slow, use carefully)"""
        try:
            domain = urlparse(url).netloc
            # Remove port if present
            domain = domain.split(':')[0]
            
            w = whois.whois(domain)
            if w.creation_date:
                creation_date = w.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                age_days = (datetime.now() - creation_date).days
                return age_days
            return -1  # Unknown
        except:
            return -1  # Error or domain doesn't exist
    
    def has_dns_record(self, url):
        """Check if domain has DNS record"""
        try:
            domain = urlparse(url).netloc
            domain = domain.split(':')[0]  # Remove port
            socket.gethostbyname(domain)
            return 1
        except:
            return 0
    
    # ========== MAIN FEATURE EXTRACTION ==========
    
    def extract_features_from_url(self, url):
        """Extract all features from a single URL"""
        features = {
            # Lexical features
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
        """
        Extract features from a DataFrame of URLs
        
        Args:
            df: DataFrame with 'url' column
            include_whois: Whether to include WHOIS features (slow!)
            sample_size: If set, only process this many URLs (for testing)
        
        Returns:
            DataFrame with extracted features
        """
        print("🔧 Extracting features from URLs...")
        print(f"   Total URLs: {len(df):,}")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            print(f"   Sampling: {len(df):,} URLs for faster processing")
        
        feature_list = []
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"   Progress: {idx:,}/{len(df):,} URLs processed...")
            
            url = row['url']
            features = self.extract_features_from_url(url)
            
            # Add WHOIS features if requested (very slow!)
            if include_whois:
                features['domain_age'] = self.get_domain_age(url)
                features['has_dns'] = self.has_dns_record(url)
            
            # Add label if present
            if 'label' in row:
                features['label'] = row['label']
            
            feature_list.append(features)
        
        print(f"   ✅ Completed: {len(feature_list):,} URLs processed")
        
        return pd.DataFrame(feature_list)

def main():
    """Main function to extract features from datasets"""
    print("🚀 Feature Extraction Module")
    print("=" * 60)
    
    # Paths
    data_dir = Path('data/processed')
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    extractor = URLFeatureExtractor()
    
    # Process each dataset
    datasets = ['train.csv', 'val.csv', 'test.csv']
    
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
            include_whois=False  # Set to True if you want domain age (very slow!)
        )
        
        # Save features
        output_path = output_dir / f'features_{dataset_name}'
        features_df.to_csv(output_path, index=False)
        print(f"   ✅ Saved to: {output_path}")
        print(f"   Features shape: {features_df.shape}")
    
    print("\n" + "="*60)
    print("✅ Feature extraction completed!")
    print(f"📁 Features saved to: {output_dir}/")
    print("\n🎯 NEXT STEPS:")
    print("   1. Explore features in notebooks/")
    print("   2. Start model training (Week 4)")
    print("="*60)

if __name__ == "__main__":
    main()