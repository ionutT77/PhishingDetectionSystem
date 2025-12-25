"""
URL Feature Extraction for Explanation Generation
Extracts characteristics from URLs to explain model predictions
"""

import re
from urllib.parse import urlparse
import tldextract


class URLFeatureExtractor:
    """Extract interpretable features from URLs for explanation generation"""
    
    def __init__(self):
        self.suspicious_keywords = [
            'login', 'verify', 'account', 'update', 'secure', 'banking',
            'paypal', 'ebay', 'signin', 'suspended', 'locked', 'confirm',
            'password', 'credential', 'authentication', 'reset', 'wallet'
        ]
        
        self.malware_keywords = [
            'download', 'install', 'setup', 'exe', 'zip', 'rar',
            'crack', 'keygen', 'patch', 'free', 'torrent'
        ]
        
        self.defacement_keywords = [
            'hacked', 'owned', 'defaced', 'pwned', 'rooted'
        ]
    
    def extract_features(self, url):
        """Extract all features from a URL"""
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        features = {
            # Basic properties
            'length': len(url),
            'domain_length': len(ext.domain) if ext.domain else 0,
            'subdomain_length': len(ext.subdomain) if ext.subdomain else 0,
            
            # Protocol
            'is_https': url.startswith('https://'),
            'has_protocol': bool(parsed.scheme),
            
            # Domain characteristics
            'has_ip': self._has_ip_address(url),
            'has_at_symbol': '@' in url,
            'domain_has_digits': bool(re.search(r'\d', ext.domain)) if ext.domain else False,
            
            # Special characters
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_question_marks': url.count('?'),
            'num_equals': url.count('='),
            'num_ampersands': url.count('&'),
            'num_percent': url.count('%'),
            
            # Suspicious patterns
            'has_suspicious_tld': self._has_suspicious_tld(ext.suffix),
            'has_multiple_subdomains': url.count('.') > 3,
            'subdomain_depth': len(ext.subdomain.split('.')) if ext.subdomain else 0,
            
            # Keywords
            'suspicious_keywords': self._find_keywords(url, self.suspicious_keywords),
            'malware_keywords': self._find_keywords(url, self.malware_keywords),
            'defacement_keywords': self._find_keywords(url, self.defacement_keywords),
            
            # Path analysis
            'path_length': len(parsed.path) if parsed.path else 0,
            'has_query': bool(parsed.query),
            'query_length': len(parsed.query) if parsed.query else 0,
            
            # Obfuscation indicators
            'has_port': bool(parsed.port),
            'port': parsed.port,
            'has_hex_chars': bool(re.search(r'%[0-9a-fA-F]{2}', url)),
            
            # Domain info
            'domain': ext.domain,
            'subdomain': ext.subdomain,
            'tld': ext.suffix,
        }
        
        return features
    
    def _has_ip_address(self, url):
        """Check if URL contains IP address instead of domain"""
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        return bool(re.search(ip_pattern, url))
    
    def _has_suspicious_tld(self, tld):
        """Check for commonly suspicious TLDs"""
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work']
        return any(tld.endswith(s) for s in suspicious_tlds)
    
    def _find_keywords(self, url, keywords):
        """Find matching keywords in URL"""
        url_lower = url.lower()
        return [kw for kw in keywords if kw in url_lower]
