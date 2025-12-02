"""
Benign URL Downloader
Downloads legitimate/benign URLs from Tranco and Cisco Umbrella top sites lists.
These lists are updated daily and contain the most popular legitimate websites.
"""

import requests
import csv
import os
import zipfile
import io
from datetime import datetime
from typing import List, Optional


class BenignURLDownloader:
    """Downloads benign URLs from popular website ranking services."""
    
    # Tranco list - research-oriented, updated daily
    TRANCO_API = "https://tranco-list.eu/top-1m.csv.zip"
    
    # Cisco Umbrella (formerly OpenDNS) - updated daily
    UMBRELLA_URL = "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
    
    # Majestic Million - updated daily
    MAJESTIC_URL = "https://downloads.majestic.com/majestic_million.csv"
    
    def __init__(self, output_dir: str = ".", source: str = "tranco"):
        """
        Initialize the benign URL downloader.
        
        Args:
            output_dir: Directory where the CSV file will be saved
            source: Source to download from ('tranco', 'umbrella', or 'majestic')
        """
        self.output_dir = output_dir
        self.source = source.lower()
        self.data: List[str] = []
        
    def download_feed(self, max_urls: int = 100000, timeout: int = 120) -> bool:
        """
        Download benign URLs from the selected source.
        
        Args:
            max_urls: Maximum number of URLs to download (default: 100,000)
            timeout: Request timeout in seconds
            
        Returns:
            True if download was successful, False otherwise
        """
        try:
            if self.source == "tranco":
                return self._download_tranco(max_urls, timeout)
            elif self.source == "umbrella":
                return self._download_umbrella(max_urls, timeout)
            elif self.source == "majestic":
                return self._download_majestic(max_urls, timeout)
            else:
                print(f"Unknown source: {self.source}")
                return False
                
        except Exception as e:
            print(f"Error downloading benign URLs: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _download_tranco(self, max_urls: int, timeout: int) -> bool:
        """Download from Tranco list."""
        print(f"Downloading benign URLs from Tranco...")
        print(f"Source: {self.TRANCO_API}")
        print("This list is updated daily and designed for security research")
        print("Downloading... (this may take a moment)")
        
        response = requests.get(self.TRANCO_API, timeout=timeout)
        response.raise_for_status()
        
        # Extract ZIP file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Get the CSV file from the ZIP
            csv_filename = zip_file.namelist()[0]
            with zip_file.open(csv_filename) as csv_file:
                # Read CSV content
                content = csv_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                # Tranco format: rank,domain
                for i, line in enumerate(lines[:max_urls]):
                    if ',' in line:
                        rank, domain = line.split(',', 1)
                        # Convert domain to full URL
                        url = f"http://{domain.strip()}"
                        self.data.append(url)
        
        print(f"Successfully downloaded {len(self.data)} benign URLs from Tranco")
        return True
    
    def _download_umbrella(self, max_urls: int, timeout: int) -> bool:
        """Download from Cisco Umbrella."""
        print(f"Downloading benign URLs from Cisco Umbrella...")
        print(f"Source: {self.UMBRELLA_URL}")
        print("This list is based on actual DNS queries and updated daily")
        print("Downloading... (this may take a moment)")
        
        response = requests.get(self.UMBRELLA_URL, timeout=timeout)
        response.raise_for_status()
        
        # Extract ZIP file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Get the CSV file from the ZIP
            csv_filename = zip_file.namelist()[0]
            with zip_file.open(csv_filename) as csv_file:
                # Read CSV content
                content = csv_file.read().decode('utf-8')
                lines = content.strip().split('\n')
                
                # Umbrella format: rank,domain
                for i, line in enumerate(lines[:max_urls]):
                    if ',' in line:
                        rank, domain = line.split(',', 1)
                        # Convert domain to full URL
                        url = f"http://{domain.strip()}"
                        self.data.append(url)
        
        print(f"Successfully downloaded {len(self.data)} benign URLs from Umbrella")
        return True
    
    def _download_majestic(self, max_urls: int, timeout: int) -> bool:
        """Download from Majestic Million."""
        print(f"Downloading benign URLs from Majestic Million...")
        print(f"Source: {self.MAJESTIC_URL}")
        print("This list is based on backlink popularity and updated daily")
        print("Downloading... (this may take a moment)")
        
        response = requests.get(self.MAJESTIC_URL, timeout=timeout)
        response.raise_for_status()
        
        # Parse CSV
        lines = response.text.strip().split('\n')
        reader = csv.DictReader(lines)
        
        count = 0
        for row in reader:
            if count >= max_urls:
                break
            
            # Majestic format has 'Domain' column
            domain = row.get('Domain', '').strip()
            if domain:
                url = f"http://{domain}"
                self.data.append(url)
                count += 1
        
        print(f"Successfully downloaded {len(self.data)} benign URLs from Majestic")
        return True
    
    def save_to_csv(self, filename: str = "Benign_URLs.csv") -> str:
        """
        Save the downloaded URLs to a CSV file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the saved CSV file
        """
        if not self.data:
            raise ValueError("No data to save. Please download the feed first.")
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                # Write in the format: url,label
                fieldnames = ['url', 'label']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for url in self.data:
                    writer.writerow({
                        'url': url,
                        'label': 'benign'
                    })
            
            print(f"\nSuccessfully saved {len(self.data)} URLs to {filepath}")
            return filepath
            
        except IOError as e:
            print(f"Error saving CSV file: {e}")
            raise
    
    def get_statistics(self) -> dict:
        """Get statistics about the downloaded URLs."""
        if not self.data:
            return {"total_urls": 0}
        
        domains = set()
        protocols = {"http": 0, "https": 0, "other": 0}
        tlds = {}
        
        for url in self.data:
            # Count protocols
            if url.startswith("https://"):
                protocols["https"] += 1
            elif url.startswith("http://"):
                protocols["http"] += 1
            else:
                protocols["other"] += 1
            
            # Extract domain and TLD
            try:
                url_without_protocol = url.split("://", 1)[1] if "://" in url else url
                domain = url_without_protocol.split("/")[0]
                domains.add(domain)
                
                # Get TLD
                if '.' in domain:
                    tld = domain.split('.')[-1]
                    tlds[tld] = tlds.get(tld, 0) + 1
            except:
                pass
        
        return {
            "total_urls": len(self.data),
            "unique_domains": len(domains),
            "protocols": protocols,
            "top_tlds": sorted(tlds.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def print_statistics(self):
        """Print statistics about the downloaded URLs."""
        stats = self.get_statistics()
        
        print("\n" + "="*50)
        print(f"Benign URLs Download Statistics ({self.source.upper()})")
        print("="*50)
        print(f"Total benign URLs: {stats['total_urls']:,}")
        
        if stats['total_urls'] > 0:
            print(f"Unique domains: {stats['unique_domains']:,}")
            print(f"\nProtocol distribution:")
            print(f"  HTTP: {stats['protocols']['http']:,}")
            print(f"  HTTPS: {stats['protocols']['https']:,}")
            
            if stats['top_tlds']:
                print(f"\nTop 10 TLDs:")
                for tld, count in stats['top_tlds']:
                    print(f"  .{tld}: {count:,}")
        
        print("="*50 + "\n")


def main():
    """Main function to download and save benign URLs."""
    print("Benign URL Downloader")
    print("-" * 50)
    print("Available sources (all updated daily):")
    print("  1. Tranco - Research-oriented ranking")
    print("  2. Umbrella - DNS query-based ranking")
    print("  3. Majestic - Backlink-based ranking")
    print()
    
    # Default to Tranco (best for security research)
    source = "tranco"
    max_urls = 100000  # Download 100k URLs by default
    
    print(f"Using source: {source.upper()}")
    print(f"Downloading up to {max_urls:,} URLs...")
    print()
    
    # Create downloader instance
    downloader = BenignURLDownloader(source=source)
    
    # Download the feed
    if downloader.download_feed(max_urls=max_urls):
        # Print statistics
        downloader.print_statistics()
        
        # Save to CSV
        try:
            filepath = downloader.save_to_csv(filename="Benign_URLs.csv")
            print(f"✓ Data successfully saved to: {filepath}")
            print(f"✓ Format: url,label")
            print(f"✓ All entries labeled as 'benign'")
            print(f"✓ Data is from today: {datetime.now().strftime('%Y-%m-%d')}")
            
        except Exception as e:
            print(f"\n✗ Failed to save data: {e}")
            return 1
    else:
        print("\n✗ Failed to download benign URLs")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
