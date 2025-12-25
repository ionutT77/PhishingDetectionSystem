"""
Download Tranco Top 1M domains and generate realistic benign URLs
Automatically downloads the latest Tranco list and creates variations
"""

import pandas as pd
import requests
from pathlib import Path
import random
from tqdm import tqdm
import zipfile
import io

print("="*70)
print("DOWNLOAD TRANCO TOP DOMAINS & GENERATE BENIGN URLs")
print("="*70)

# Download Tranco list
print("\n📥 Downloading Tranco Top 1M domains...")

try:
    # Get latest Tranco list
    response = requests.get('https://tranco-list.eu/top-1m.csv.zip', timeout=30)
    response.raise_for_status()
    
    # Extract ZIP
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            tranco_df = pd.read_csv(f, names=['rank', 'domain'])
    
    print(f"✅ Downloaded {len(tranco_df):,} domains from Tranco")
    
except Exception as e:
    print(f"❌ Failed to download Tranco list: {e}")
    print("\n💡 Using backup: Top 10,000 domains (manual list)")
    
    # Backup: manually curated top domains
    top_domains = [
        'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'reddit.com', 'wikipedia.org', 'amazon.com', 'yahoo.com',
        'ebay.com', 'apple.com', 'microsoft.com', 'github.com', 'stackoverflow.com',
        # ... (we'll use these if download fails)
    ]
    tranco_df = pd.DataFrame({'rank': range(1, len(top_domains)+1), 'domain': top_domains})

# URL path variations (realistic common paths)
URL_PATHS = [
    '', '/', '/home', '/index.html', '/index.php',
    '/about', '/about-us', '/company', '/contact', '/help',
    '/login', '/signin', '/signup', '/register', '/account',
    '/products', '/services', '/solutions', '/pricing', '/features',
    '/blog', '/news', '/articles', '/docs', '/api',
    '/support', '/faq', '/terms', '/privacy', '/legal',
    '/search', '/download', '/downloads', '/media', '/images',
    '/en/', '/en-us/', '/shop', '/store', '/cart',
    '/profile', '/settings', '/dashboard', '/admin',
]

# Subdomains (common legitimate ones)
SUBDOMAINS = ['', 'www', 'blog', 'shop', 'mail', 'support', 'api', 'docs', 'cdn', 'static']

# Protocols
PROTOCOLS = ['https://', 'http://']

def generate_url_variations(domain, num_variations=5):
    """Generate realistic URL variations for a domain"""
    urls = []
    
    for _ in range(num_variations):
        # Protocol (90% HTTPS, 10% HTTP for realism)
        protocol = random.choices(PROTOCOLS, weights=[0.9, 0.1])[0]
        
        # Subdomain (60% no subdomain, 40% with subdomain)
        subdomain = random.choice(SUBDOMAINS) if random.random() < 0.4 else ''
        full_domain = f"{subdomain}.{domain}" if subdomain else domain
        
        # Path (70% with path, 30% just domain)
        path = random.choice(URL_PATHS) if random.random() < 0.7 else ''
        
        url = f"{protocol}{full_domain}{path}"
        urls.append(url)
    
    return urls

# Generate URLs from top domains
print("\n🔧 Generating URL variations...")

# Take top 100k domains and generate 3-5 URLs each = ~400k URLs
TOP_N = 100000
VARIATIONS_PER_DOMAIN = 4

all_urls = []
domains_to_process = tranco_df.head(TOP_N)

for _, row in tqdm(domains_to_process.iterrows(), total=len(domains_to_process), desc="Processing domains"):
    domain = row['domain']
    urls = generate_url_variations(domain, num_variations=VARIATIONS_PER_DOMAIN)
    all_urls.extend(urls)

# Remove duplicates
print(f"\n🔍 Removing duplicates...")
initial_count = len(all_urls)
all_urls = list(set(all_urls))  # Remove duplicates
final_count = len(all_urls)
duplicates_removed = initial_count - final_count

print(f"   Initial URLs: {initial_count:,}")
print(f"   Unique URLs:  {final_count:,}")
print(f"   Duplicates removed: {duplicates_removed:,}")

# Create DataFrame
df = pd.DataFrame({
    'url': all_urls,
    'label': 'benign'
})

# Save
output_file = Path('../../data/tranco_benign_urls.csv')
df.to_csv(output_file, index=False)

print(f"\n💾 Saved {len(df):,} benign URLs to: {output_file}")

# Statistics
print(f"\n📊 Statistics:")
https_count = sum(1 for url in all_urls if url.startswith('https://'))
http_count = len(all_urls) - https_count
print(f"   HTTPS: {https_count:,} ({https_count/len(all_urls)*100:.1f}%)")
print(f"   HTTP:  {http_count:,} ({http_count/len(all_urls)*100:.1f}%)")

with_subdomain = sum(1 for url in all_urls if url.count('.') > 1 and 'www' not in url)
print(f"   With subdomain: {with_subdomain:,} ({with_subdomain/len(all_urls)*100:.1f}%)")

with_path = sum(1 for url in all_urls if url.count('/') > 2)
print(f"   With path: {with_path:,} ({with_path/len(all_urls)*100:.1f}%)")

# Sample URLs
print(f"\n🔍 Sample URLs:")
for i, url in enumerate(random.sample(all_urls, min(15, len(all_urls))), 1):
    print(f"   {i:2}. {url}")

print("\n" + "="*70)
print("✅ DOWNLOAD COMPLETE!")
print("="*70)
print(f"\n🚀 Next step: Run merge_tranco_with_dataset.py")
print("="*70)
