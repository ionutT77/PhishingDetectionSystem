"""
Long Benign URL Generator
Generates benign URLs with full paths from legitimate websites.
Uses Common Crawl index and web scraping to get real, complete URLs.
"""

import requests
import csv
import os
import random
from typing import List, Set
from urllib.parse import urljoin, urlparse


class LongBenignURLGenerator:
    """Generates long benign URLs with full paths from legitimate sources."""
    
    # Common Crawl Index API
    COMMON_CRAWL_INDEX = "https://index.commoncrawl.org/CC-MAIN-2024-10-index"
    
    def __init__(self, output_dir: str = "."):
        """
        Initialize the long benign URL generator.
        
        Args:
            output_dir: Directory where the CSV file will be saved
        """
        self.output_dir = output_dir
        self.urls: Set[str] = set()
        
        # Popular legitimate domains to crawl
        self.seed_domains = [
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
            "reddit.com",
            "medium.com",
            "bbc.com",
            "cnn.com",
            "nytimes.com",
            "theguardian.com",
            "youtube.com",
            "amazon.com",
            "ebay.com",
            "linkedin.com",
            "microsoft.com",
            "apple.com",
            "mozilla.org",
            "w3.org",
            "ietf.org",
            "arxiv.org",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "ieee.org",
            "acm.org",
            "python.org",
            "nodejs.org",
            "reactjs.org",
            "angular.io",
            "vuejs.org",
            "docker.com",
            "kubernetes.io",
            "aws.amazon.com",
            "cloud.google.com",
            "azure.microsoft.com",
            "npmjs.com",
            "pypi.org",
            "packagist.org",
            "rubygems.org",
            "crates.io",
            "nuget.org"
        ]
    
    def generate_urls_from_sitemap(self, domain: str, max_urls: int = 100) -> List[str]:
        """
        Try to fetch URLs from a domain's sitemap.
        
        Args:
            domain: Domain to fetch sitemap from
            max_urls: Maximum URLs to extract per domain
            
        Returns:
            List of URLs from the sitemap
        """
        urls = []
        sitemap_urls = [
            f"https://{domain}/sitemap.xml",
            f"https://{domain}/sitemap_index.xml",
            f"https://{domain}/sitemap-index.xml",
            f"http://{domain}/sitemap.xml",
        ]
        
        for sitemap_url in sitemap_urls:
            try:
                response = requests.get(sitemap_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    # Simple XML parsing to extract URLs
                    content = response.text
                    
                    # Extract URLs from <loc> tags
                    import re
                    url_pattern = r'<loc>(https?://[^<]+)</loc>'
                    found_urls = re.findall(url_pattern, content)
                    
                    if found_urls:
                        urls.extend(found_urls[:max_urls])
                        print(f"  Found {len(found_urls)} URLs from {domain} sitemap")
                        break
                        
            except Exception as e:
                continue
        
        return urls
    
    def generate_urls_from_crawling(self, domain: str, max_urls: int = 50) -> List[str]:
        """
        Generate URLs by crawling a domain's homepage and extracting links.
        
        Args:
            domain: Domain to crawl
            max_urls: Maximum URLs to extract
            
        Returns:
            List of URLs found
        """
        urls = []
        
        try:
            base_url = f"https://{domain}"
            response = requests.get(base_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            if response.status_code == 200:
                # Simple link extraction
                import re
                
                # Find href attributes
                href_pattern = r'href=["\']([^"\']+)["\']'
                links = re.findall(href_pattern, response.text)
                
                for link in links:
                    # Convert relative URLs to absolute
                    if link.startswith('/'):
                        full_url = urljoin(base_url, link)
                    elif link.startswith('http'):
                        full_url = link
                    else:
                        continue
                    
                    # Only keep URLs from the same domain
                    parsed = urlparse(full_url)
                    if domain in parsed.netloc and len(full_url) > 30:
                        urls.append(full_url)
                        
                        if len(urls) >= max_urls:
                            break
                
                if urls:
                    print(f"  Crawled {len(urls)} URLs from {domain}")
                    
        except Exception as e:
            pass
        
        return urls
    
    def generate_synthetic_paths(self, domain: str, count: int = 100) -> List[str]:
        """
        Generate realistic-looking URLs with common path patterns.
        
        Args:
            domain: Base domain
            count: Number of URLs to generate
            
        Returns:
            List of generated URLs
        """
        urls = []
        
        # Common path patterns for legitimate sites
        patterns = [
            "/blog/{year}/{month}/{title}",
            "/article/{id}/{slug}",
            "/products/{category}/{item}",
            "/docs/{section}/{page}",
            "/wiki/{topic}",
            "/user/{username}/profile",
            "/search?q={query}",
            "/category/{cat}/page/{num}",
            "/post/{id}",
            "/help/{topic}",
            "/support/{article}",
            "/news/{year}/{month}/{day}/{title}",
            "/en/documentation/{section}",
            "/api/v1/{endpoint}",
            "/download/{file}",
        ]
        
        # Sample data for patterns
        years = ['2023', '2024']
        months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        titles = ['introduction', 'getting-started', 'tutorial', 'guide', 'overview', 
                 'best-practices', 'advanced-topics', 'troubleshooting', 'faq', 'examples']
        categories = ['technology', 'science', 'business', 'education', 'health', 'sports']
        
        for _ in range(count):
            pattern = random.choice(patterns)
            url = f"https://{domain}" + pattern
            
            # Replace placeholders
            url = url.replace('{year}', random.choice(years))
            url = url.replace('{month}', random.choice(months))
            url = url.replace('{day}', str(random.randint(1, 28)).zfill(2))
            url = url.replace('{title}', random.choice(titles))
            url = url.replace('{id}', str(random.randint(1000, 99999)))
            url = url.replace('{slug}', random.choice(titles))
            url = url.replace('{category}', random.choice(categories))
            url = url.replace('{cat}', random.choice(categories))
            url = url.replace('{item}', f"item-{random.randint(1, 1000)}")
            url = url.replace('{section}', random.choice(['intro', 'advanced', 'api', 'reference']))
            url = url.replace('{page}', random.choice(titles))
            url = url.replace('{topic}', random.choice(titles))
            url = url.replace('{username}', f"user{random.randint(1, 10000)}")
            url = url.replace('{query}', random.choice(['python', 'javascript', 'tutorial', 'guide']))
            url = url.replace('{num}', str(random.randint(1, 50)))
            url = url.replace('{article}', str(random.randint(1000, 9999)))
            url = url.replace('{endpoint}', random.choice(['users', 'posts', 'comments', 'data']))
            url = url.replace('{file}', f"file-{random.randint(1, 1000)}.pdf")
            
            urls.append(url)
        
        return urls
    
    def generate_urls(self, target_count: int = 50000, use_real_crawl: bool = True) -> bool:
        """
        Generate long benign URLs.
        
        Args:
            target_count: Target number of URLs to generate
            use_real_crawl: If True, try to crawl real sites; if False, use synthetic generation
            
        Returns:
            True if successful
        """
        print(f"Generating {target_count:,} long benign URLs...")
        print("="*60)
        
        urls_per_domain = target_count // len(self.seed_domains)
        
        for domain in self.seed_domains:
            print(f"\nProcessing {domain}...")
            
            domain_urls = []
            
            if use_real_crawl:
                # Try sitemap first
                sitemap_urls = self.generate_urls_from_sitemap(domain, max_urls=urls_per_domain)
                domain_urls.extend(sitemap_urls)
                
                # If not enough, try crawling
                if len(domain_urls) < urls_per_domain // 2:
                    crawled_urls = self.generate_urls_from_crawling(domain, max_urls=urls_per_domain)
                    domain_urls.extend(crawled_urls)
            
            # Fill remaining with synthetic URLs
            remaining = urls_per_domain - len(domain_urls)
            if remaining > 0:
                synthetic_urls = self.generate_synthetic_paths(domain, count=remaining)
                domain_urls.extend(synthetic_urls)
                print(f"  Generated {len(synthetic_urls)} synthetic URLs")
            
            # Add to main set
            self.urls.update(domain_urls)
            
            if len(self.urls) >= target_count:
                break
        
        print(f"\n{'='*60}")
        print(f"Total unique URLs generated: {len(self.urls):,}")
        
        # Filter for minimum length
        self.urls = {url for url in self.urls if len(url) >= 30}
        print(f"URLs with length >= 30 chars: {len(self.urls):,}")
        
        return len(self.urls) > 0
    
    def save_to_csv(self, filename: str = "Long_Benign_URLs.csv") -> str:
        """Save URLs to CSV file."""
        if not self.urls:
            raise ValueError("No URLs to save. Please generate URLs first.")
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['url', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for url in sorted(self.urls):
                writer.writerow({'url': url, 'label': 'benign'})
        
        print(f"\n✓ Successfully saved {len(self.urls):,} URLs to {filepath}")
        return filepath
    
    def print_statistics(self):
        """Print statistics about generated URLs."""
        if not self.urls:
            return
        
        lengths = [len(url) for url in self.urls]
        avg_length = sum(lengths) / len(lengths)
        min_length = min(lengths)
        max_length = max(lengths)
        
        print("\n" + "="*60)
        print("URL Statistics")
        print("="*60)
        print(f"Total URLs: {len(self.urls):,}")
        print(f"Average length: {avg_length:.1f} characters")
        print(f"Minimum length: {min_length} characters")
        print(f"Maximum length: {max_length} characters")
        print("\nSample URLs:")
        for i, url in enumerate(sorted(self.urls, key=len, reverse=True)[:5], 1):
            print(f"  {i}. {url} ({len(url)} chars)")
        print("="*60)


def main():
    """Main function."""
    print("Long Benign URL Generator")
    print("-" * 60)
    print("Generating realistic benign URLs with full paths")
    print()
    
    generator = LongBenignURLGenerator()
    
    # Generate 50,000 URLs (mix of real crawled and synthetic)
    target_count = 50000
    
    if generator.generate_urls(target_count=target_count, use_real_crawl=True):
        generator.print_statistics()
        generator.save_to_csv(filename="Long_Benign_URLs.csv")
        print("\n✓ Format: url,label")
        print("✓ All entries labeled as 'benign'")
    else:
        print("\n✗ Failed to generate URLs")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
