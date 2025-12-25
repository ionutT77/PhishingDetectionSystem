"""
Generate 300k+ Benign URLs from Top Legitimate Domains
Creates variations with paths, subdomains, query parameters, etc.
"""

import pandas as pd
import random
from pathlib import Path

# Top 100 legitimate domains (most trusted websites worldwide)
TOP_DOMAINS = [
    # Search engines & tech giants
    'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'instagram.com',
    'linkedin.com', 'reddit.com', 'wikipedia.org', 'amazon.com', 'ebay.com',
    'apple.com', 'microsoft.com', 'github.com', 'stackoverflow.com', 'medium.com',
    
    # News & media
    'nytimes.com', 'cnn.com', 'bbc.com', 'theguardian.com', 'forbes.com',
    'reuters.com', 'bloomberg.com', 'wsj.com', 'washingtonpost.com', 'usatoday.com',
    
    # E-commerce
    'shopify.com', 'etsy.com', 'walmart.com', 'target.com', 'bestbuy.com',
    'aliexpress.com', 'alibaba.com', 'ebay.com', 'craigslist.org', 'booking.com',
    
    # Cloud & hosting
    'cloudflare.com', 'dropbox.com', 'drive.google.com', 'onedrive.com', 'icloud.com',
    'aws.amazon.com', 'azure.microsoft.com', 'heroku.com', 'netlify.com', 'vercel.com',
    
    # Development & education
    'npmjs.com', 'pypi.org', 'docker.com', 'gitlab.com', 'bitbucket.org',
    'coursera.org', 'udemy.com', 'khanacademy.org', 'edx.org', 'mit.edu',
    
    # Government & organizations
    'gov.uk', 'canada.ca', 'nih.gov', 'cdc.gov', 'who.int',
    'un.org', 'europa.eu', 'nasa.gov', 'noaa.gov', 'weather.com',
    
    # Entertainment
    'netflix.com', 'spotify.com', 'hulu.com', 'twitch.tv', 'vimeo.com',
    'imdb.com', 'rottentomatoes.com', 'metacritic.com', 'gamespot.com', 'ign.com',
    
    # Finance & banking
    'paypal.com', 'stripe.com', 'chase.com', 'bankofamerica.com', 'wellsfargo.com',
    'citibank.com', 'capitalone.com', 'coinbase.com', 'binance.com', 'kraken.com',
    
    # Social & communication
    'discord.com', 'slack.com', 'zoom.us', 'teams.microsoft.com', 'whatsapp.com',
    'telegram.org', 'signal.org', 'skype.com', 'snapchat.com', 'tiktok.com',
    
    # Additional trusted domains
    'adobe.com', 'salesforce.com', 'oracle.com', 'ibm.com', 'intel.com',
    'nvidia.com', 'amd.com', 'samsung.com', 'sony.com', 'lg.com'
]

# Common URL paths (categorized for realism)
PATHS = {
    'navigation': [
        '', 'home', 'index.html', 'index.php', 'main',
        'dashboard', 'portal', 'app', 'web'
    ],
    'info': [
        'about', 'about-us', 'company', 'team', 'careers',
        'contact', 'contact-us', 'support', 'help', 'faq',
        'privacy', 'terms', 'legal', 'sitemap', 'blog'
    ],
    'user': [
        'login', 'signin', 'signup', 'register', 'account',
        'profile', 'settings', 'preferences', 'logout', 'auth',
        'user/dashboard', 'user/profile', 'user/settings'
    ],
    'content': [
        'products', 'services', 'solutions', 'features', 'pricing',
        'news', 'blog', 'articles', 'docs', 'documentation',
        'api', 'developers', 'resources', 'guides', 'tutorials'
    ],
    'actions': [
        'search', 'download', 'upload', 'share', 'subscribe',
        'checkout', 'cart', 'wishlist', 'compare', 'review'
    ],
    'media': [
        'images', 'videos', 'gallery', 'photos', 'media',
        'downloads', 'files', 'assets', 'cdn', 'static'
    ]
}

# Common subdomains
SUBDOMAINS = [
    '', 'www', 'blog', 'shop', 'store', 'api', 'dev', 'docs',
    'support', 'help', 'mail', 'admin', 'dashboard', 'app',
    'mobile', 'secure', 'my', 'account', 'cdn', 'static',
    'm', 'beta', 'test', 'staging', 'portal', 'forum'
]

# Query parameters (realistic combinations)
QUERY_PARAMS = [
    '',
    '?page=1', '?page=2', '?page=home',
    '?id=123', '?id=456', '?user=admin',
    '?q=search', '?q=help', '?search=product',
    '?lang=en', '?lang=es', '?locale=en_US',
    '?ref=home', '?ref=email', '?source=google',
    '?utm_source=google&utm_medium=cpc',
    '?category=technology', '?tag=news',
    '?sort=date', '?order=asc', '?limit=10',
    '?tab=overview', '?view=grid', '?mode=light'
]

# File extensions (for download/resource URLs)
EXTENSIONS = [
    '', '.html', '.php', '.asp', '.aspx', '.jsp',
    '.pdf', '.jpg', '.png', '.gif', '.svg',
    '.css', '.js', '.json', '.xml'
]

# Protocols
PROTOCOLS = ['https://', 'http://']

def generate_url(domain):
    """Generate a random realistic URL variation"""
    
    # Choose protocol (80% https, 20% http for realism)
    protocol = random.choices(PROTOCOLS, weights=[0.8, 0.2])[0]
    
    # Choose subdomain (50% no subdomain, 50% with subdomain)
    subdomain = random.choice(SUBDOMAINS)
    if subdomain:
        full_domain = f"{subdomain}.{domain}"
    else:
        full_domain = domain
    
    # Choose path category and path
    category = random.choice(list(PATHS.keys()))
    path = random.choice(PATHS[category])
    
    # Add extension sometimes
    if path and random.random() < 0.2:
        extension = random.choice(EXTENSIONS)
        path = f"{path}{extension}"
    
    # Add query parameters (30% of URLs)
    query = ''
    if random.random() < 0.3:
        query = random.choice(QUERY_PARAMS)
    
    # Add anchor (5% of URLs)
    anchor = ''
    if random.random() < 0.05:
        anchors = ['top', 'content', 'footer', 'section1', 'features', 'pricing']
        anchor = f"#{random.choice(anchors)}"
    
    # Construct URL
    if path:
        url = f"{protocol}{full_domain}/{path}{query}{anchor}"
    else:
        url = f"{protocol}{full_domain}{query}{anchor}"
    
    return url

def generate_benign_dataset(target_count=300000, output_file='benign_urls_generated.csv'):
    """Generate target_count benign URLs"""
    
    print("="*70)
    print(f"GENERATING {target_count:,} BENIGN URLs")
    print("="*70)
    
    print(f"\n🌐 Using {len(TOP_DOMAINS)} top legitimate domains")
    print(f"🔀 Generating variations with:")
    print(f"   • {len(SUBDOMAINS)} subdomains")
    print(f"   • {sum(len(p) for p in PATHS.values())} paths")
    print(f"   • {len(QUERY_PARAMS)} query parameter patterns")
    print(f"   • {len(PROTOCOLS)} protocols")
    
    urls = []
    urls_set = set()  # To avoid duplicates
    
    print(f"\n🔧 Generating URLs...")
    
    # Generate URLs in batches for progress tracking
    batch_size = 10000
    batches = (target_count + batch_size - 1) // batch_size
    
    for batch in range(batches):
        batch_urls = []
        attempts = 0
        max_attempts = batch_size * 10  # Safety limit
        
        while len(batch_urls) < batch_size and attempts < max_attempts:
            domain = random.choice(TOP_DOMAINS)
            url = generate_url(domain)
            
            # Avoid duplicates
            if url not in urls_set:
                urls_set.add(url)
                batch_urls.append(url)
            
            attempts += 1
        
        urls.extend(batch_urls)
        
        if (batch + 1) % 10 == 0 or batch == batches - 1:
            print(f"   Progress: {len(urls):,}/{target_count:,} URLs generated")
        
        # Stop if we've reached target
        if len(urls) >= target_count:
            break
    
    # Trim to exact count
    urls = urls[:target_count]
    
    # Create DataFrame
    df = pd.DataFrame({
        'url': urls,
        'label': 'benign'
    })
    
    # Save to CSV
    output_path = Path(output_file)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Generated {len(urls):,} unique benign URLs")
    print(f"💾 Saved to: {output_path.absolute()}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    https_count = sum(1 for url in urls if url.startswith('https://'))
    http_count = len(urls) - https_count
    print(f"   HTTPS: {https_count:,} ({https_count/len(urls)*100:.1f}%)")
    print(f"   HTTP:  {http_count:,} ({http_count/len(urls)*100:.1f}%)")
    
    with_subdomain = sum(1 for url in urls if url.count('.') > 1)
    print(f"   With subdomain: {with_subdomain:,} ({with_subdomain/len(urls)*100:.1f}%)")
    
    with_params = sum(1 for url in urls if '?' in url)
    print(f"   With query params: {with_params:,} ({with_params/len(urls)*100:.1f}%)")
    
    # Sample URLs
    print(f"\n🔍 Sample URLs:")
    for i, url in enumerate(random.sample(urls, min(10, len(urls))), 1):
        print(f"   {i:2}. {url}")
    
    return df

if __name__ == "__main__":
    # Generate 300k benign URLs
    df = generate_benign_dataset(
        target_count=300000,
        output_file='../../data/benign_urls_generated_300k.csv'
    )
    
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE!")
    print("="*70)
    print("\n🚀 Next steps:")
    print("   1. Run: python merge_benign_urls.py")
    print("      (This will merge with existing dataset)")
    print("   2. Upload new balanced dataset to Kaggle")
    print("   3. Retrain model with balanced data")
    print("="*70)
