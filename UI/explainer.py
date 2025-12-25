"""
URL Prediction Explainer
Generates human-readable explanations for phishing detection predictions
"""

from feature_extractor import URLFeatureExtractor


class URLExplainer:
    """Generate explanations for URL classification predictions"""
    
    def __init__(self):
        self.extractor = URLFeatureExtractor()
    
    def generate_explanation(self, url, predicted_class, probabilities):
        """
        Generate a detailed explanation for the prediction
        
        Args:
            url: The URL that was classified
            predicted_class: The predicted class name
            probabilities: Dictionary of class probabilities
        
        Returns:
            Dictionary with explanation components
        """
        features = self.extractor.extract_features(url)
        
        explanation = {
            'primary_factors': [],
            'risk_indicators': [],
            'safe_indicators': [],
            'technical_details': {},
            'confidence_explanation': ''
        }
        
        # Identify risk factors based on features
        risk_factors = self._identify_risk_factors(features, predicted_class)
        safe_factors = self._identify_safe_factors(features, predicted_class)
        
        explanation['risk_indicators'] = risk_factors
        explanation['safe_indicators'] = safe_factors
        
        # Generate primary explanation based on predicted class
        if predicted_class == 'phishing':
            explanation['primary_factors'] = self._explain_phishing(features)
        elif predicted_class == 'malware':
            explanation['primary_factors'] = self._explain_malware(features)
        elif predicted_class == 'defacement':
            explanation['primary_factors'] = self._explain_defacement(features)
        else:  # benign
            explanation['primary_factors'] = self._explain_benign(features)
        
        # Technical details
        explanation['technical_details'] = {
            'URL Length': features['length'],
            'Domain': features['domain'] or 'Unknown',
            'Protocol': 'HTTPS (Secure)' if features['is_https'] else 'HTTP (Insecure)',
            'Has IP Address': 'Yes ⚠️' if features['has_ip'] else 'No',
            'Subdomain Depth': features['subdomain_depth'],
        }
        
        # Confidence explanation
        max_prob = max(probabilities.values())
        explanation['confidence_explanation'] = self._explain_confidence(max_prob, predicted_class)
        
        return explanation
    
    def _identify_risk_factors(self, features, predicted_class):
        """Identify risk indicators in the URL"""
        risks = []
        
        # Length-based risks
        if features['length'] > 100:
            risks.append(f"Very long URL ({features['length']} characters) - often used to hide malicious intent")
        
        # Protocol risks
        if not features['is_https']:
            risks.append("Uses HTTP instead of HTTPS - connection is not encrypted")
        
        # Domain risks
        if features['has_ip']:
            risks.append("Uses IP address instead of domain name - highly suspicious")
        
        if features['has_at_symbol']:
            risks.append("Contains @ symbol - can be used to disguise real destination")
        
        if features['domain_has_digits']:
            risks.append("Domain contains numbers - uncommon for legitimate sites")
        
        # Subdomain risks
        if features['subdomain_depth'] > 2:
            risks.append(f"Deep subdomain structure ({features['subdomain_depth']} levels) - potential spoofing")
        
        if features['has_multiple_subdomains']:
            risks.append("Multiple subdomains detected - common in phishing attacks")
        
        # TLD risks
        if features['has_suspicious_tld']:
            risks.append(f"Suspicious top-level domain (.{features['tld']}) - commonly used by attackers")
        
        # Keyword risks
        if features['suspicious_keywords']:
            keywords = ', '.join(features['suspicious_keywords'][:3])
            risks.append(f"Suspicious keywords detected: {keywords}")
        
        if features['malware_keywords']:
            keywords = ', '.join(features['malware_keywords'][:3])
            risks.append(f"Malware-related keywords: {keywords}")
        
        if features['defacement_keywords']:
            keywords = ', '.join(features['defacement_keywords'][:3])
            risks.append(f"Defacement indicators: {keywords}")
        
        # Obfuscation risks
        if features['has_hex_chars']:
            risks.append("URL encoding detected - may hide malicious content")
        
        if features['num_percent'] > 3:
            risks.append(f"Excessive URL encoding ({features['num_percent']} % symbols)")
        
        # Special character risks
        if features['num_hyphens'] > 3:
            risks.append(f"Many hyphens ({features['num_hyphens']}) - typosquatting indicator")
        
        return risks
    
    def _identify_safe_factors(self, features, predicted_class):
        """Identify safety indicators in the URL"""
        safe = []
        
        if features['is_https']:
            safe.append("Uses HTTPS protocol - encrypted connection")
        
        if not features['has_ip']:
            safe.append("Uses domain name instead of IP address")
        
        if features['length'] < 50:
            safe.append("Short URL length - easier to verify")
        
        if features['subdomain_depth'] <= 1:
            safe.append("Simple domain structure - less suspicious")
        
        if not features['suspicious_keywords'] and not features['malware_keywords']:
            safe.append("No suspicious keywords detected")
        
        if features['domain'] and len(features['domain']) > 3:
            safe.append(f"Readable domain name: {features['domain']}")
        
        return safe
    
    def _explain_phishing(self, features):
        """Generate explanation for phishing classification"""
        explanations = [
            "The URL exhibits patterns commonly associated with phishing attacks:"
        ]
        
        if features['suspicious_keywords']:
            explanations.append(f"• Contains phishing keywords: {', '.join(features['suspicious_keywords'][:3])}")
        
        if features['length'] > 80:
            explanations.append(f"• Unusually long URL ({features['length']} chars) to obscure real destination")
        
        if features['has_ip']:
            explanations.append("• Uses IP address instead of legitimate domain")
        
        if features['subdomain_depth'] > 1:
            explanations.append(f"• Complex subdomain structure to mimic legitimate sites")
        
        return explanations
    
    def _explain_malware(self, features):
        """Generate explanation for malware classification"""
        explanations = [
            "The URL shows characteristics typical of malware distribution:"
        ]
        
        if features['malware_keywords']:
            explanations.append(f"• Contains malware-related terms: {', '.join(features['malware_keywords'][:3])}")
        
        if not features['is_https']:
            explanations.append("• Insecure HTTP connection - easier to inject malicious code")
        
        if features['has_suspicious_tld']:
            explanations.append(f"• Suspicious TLD (.{features['tld']}) frequently used for malware")
        
        return explanations
    
    def _explain_defacement(self, features):
        """Generate explanation for defacement classification"""
        explanations = [
            "The URL matches patterns associated with defaced websites:"
        ]
        
        if features['defacement_keywords']:
            explanations.append(f"• Contains defacement indicators: {', '.join(features['defacement_keywords'][:3])}")
        
        if features['has_ip']:
            explanations.append("• Direct IP access - bypasses normal domain security")
        
        return explanations
    
    def _explain_benign(self, features):
        """Generate explanation for benign classification"""
        explanations = [
            "The URL appears to be legitimate based on:"
        ]
        
        if features['is_https']:
            explanations.append("• Secure HTTPS protocol")
        
        if features['domain']:
            explanations.append(f"• Recognizable domain: {features['domain']}")
        
        if not features['has_ip']:
            explanations.append("• Uses proper domain name instead of IP")
        
        if features['length'] < 60:
            explanations.append("• Reasonable URL length")
        
        return explanations
    
    def _explain_confidence(self, probability, predicted_class):
        """Explain the confidence level of the prediction"""
        confidence_pct = probability * 100
        
        if confidence_pct >= 90:
            return f"Very high confidence ({confidence_pct:.1f}%) - strong indicators of {predicted_class}"
        elif confidence_pct >= 75:
            return f"High confidence ({confidence_pct:.1f}%) - multiple indicators point to {predicted_class}"
        elif confidence_pct >= 60:
            return f"Moderate confidence ({confidence_pct:.1f}%) - some indicators suggest {predicted_class}"
        else:
            return f"Low confidence ({confidence_pct:.1f}%) - weak indicators, uncertain classification"
