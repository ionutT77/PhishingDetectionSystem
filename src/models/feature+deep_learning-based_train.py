"""
Hybrid Phishing Detection Model
Combines hand-crafted features with deep learning on raw URLs
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib

class FeatureExtractor:
    """Extract hand-crafted features from URLs"""
    
    @staticmethod
    def extract_features(url):
        """Extract URL features"""
        import re
        from urllib.parse import urlparse
        
        features = {}
        
        # Length features
        features['url_length'] = len(url)
        features['domain_length'] = len(urlparse(url).netloc)
        features['path_length'] = len(urlparse(url).path)
        
        # Character counts
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_questionmarks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_digits'] = sum(c.isdigit() for c in url)
        
        # Ratios
        if len(url) > 0:
            features['digit_ratio'] = features['num_digits'] / len(url)
            features['letter_ratio'] = sum(c.isalpha() for c in url) / len(url)
        else:
            features['digit_ratio'] = 0
            features['letter_ratio'] = 0
        
        # Boolean features
        features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        features['has_https'] = 1 if url.startswith('https://') else 0
        features['has_www'] = 1 if 'www.' in url else 0
        
        # Subdomain count
        domain = urlparse(url).netloc
        features['num_subdomains'] = domain.count('.') - 1 if domain else 0
        
        return features

class HybridPhishingDetector:
    """Combines features and deep learning"""
    
    def __init__(self, max_url_length=200):
        self.max_url_length = max_url_length
        self.char_to_idx = {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def build_vocabulary(self, urls):
        """Build character vocabulary"""
        print("📚 Building character vocabulary...")
        all_chars = set()
        for url in urls:
            all_chars.update(url)
        
        self.char_to_idx = {char: idx + 2 for idx, char in enumerate(sorted(all_chars))}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 1
        
        print(f"   Vocabulary size: {len(self.char_to_idx)}")
        return self
    
    def encode_url(self, url):
        """Encode URL to sequence"""
        url = url[:self.max_url_length]
        encoded = [self.char_to_idx.get(char, 1) for char in url]
        if len(encoded) < self.max_url_length:
            encoded += [0] * (self.max_url_length - len(encoded))
        return encoded
    
    def extract_features(self, urls):
        """Extract hand-crafted features"""
        print("🔧 Extracting hand-crafted features...")
        features_list = []
        for url in urls:
            features_list.append(FeatureExtractor.extract_features(url))
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = list(features_df.columns)
        return features_df.values
    
    def prepare_data(self, urls, is_training=False):
        """Prepare both URL sequences and features"""
        # Encode URLs
        url_sequences = np.array([self.encode_url(url) for url in urls])
        
        # Extract features
        extracted_features = self.extract_features(urls)
        
        # Scale features
        if is_training:
            scaled_features = self.scaler.fit_transform(extracted_features)
        else:
            scaled_features = self.scaler.transform(extracted_features)
        
        return url_sequences, scaled_features
    
    def build_model(self, num_features):
        """Build hybrid model architecture"""
        print("\n🏗️  Building hybrid neural network...")
        
        # Input 1: URL sequence
        url_input = layers.Input(shape=(self.max_url_length,), name='url_input')
        
        # URL processing branch
        url_embedding = layers.Embedding(
            input_dim=len(self.char_to_idx),
            output_dim=64
        )(url_input)
        
        url_conv1 = layers.Conv1D(128, 3, activation='relu', padding='same')(url_embedding)
        url_pool1 = layers.MaxPooling1D(2)(url_conv1)
        url_drop1 = layers.Dropout(0.3)(url_pool1)
        
        url_conv2 = layers.Conv1D(128, 3, activation='relu', padding='same')(url_drop1)
        url_pool2 = layers.MaxPooling1D(2)(url_conv2)
        url_drop2 = layers.Dropout(0.3)(url_pool2)
        
        url_lstm = layers.Bidirectional(layers.LSTM(64))(url_drop2)
        url_drop3 = layers.Dropout(0.3)(url_lstm)
        
        # Input 2: Hand-crafted features
        feature_input = layers.Input(shape=(num_features,), name='feature_input')
        
        # Feature processing branch
        feature_dense1 = layers.Dense(64, activation='relu')(feature_input)
        feature_drop1 = layers.Dropout(0.3)(feature_dense1)
        feature_dense2 = layers.Dense(32, activation='relu')(feature_drop1)
        feature_drop2 = layers.Dropout(0.2)(feature_dense2)
        
        # Combine both branches
        combined = layers.concatenate([url_drop3, feature_drop2])
        
        # Final classification layers
        dense1 = layers.Dense(64, activation='relu')(combined)
        drop1 = layers.Dropout(0.3)(dense1)
        dense2 = layers.Dense(32, activation='relu')(drop1)
        drop2 = layers.Dropout(0.2)(dense2)
        
        output = layers.Dense(1, activation='sigmoid', name='output')(drop2)
        
        # Create model
        model = keras.Model(
            inputs=[url_input, feature_input],
            outputs=output
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        
        print("\n📊 Model Architecture:")
        model.summary()
        
        return self
    
    def train(self, urls_train, y_train, urls_val, y_val, epochs=30, batch_size=32):
        """Train hybrid model"""
        print("\n🚀 Starting hybrid model training...")
        print("="*60)
        
        # Prepare training data
        X_url_train, X_feat_train = self.prepare_data(urls_train, is_training=True)
        X_url_val, X_feat_val = self.prepare_data(urls_val, is_training=False)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'models/hybrid_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            [X_url_train, X_feat_train],
            y_train,
            validation_data=([X_url_val, X_feat_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, urls, y_true, dataset_name="Test"):
        """Evaluate model"""
        print(f"\n📊 Evaluating on {dataset_name} set...")
        
        X_url, X_feat = self.prepare_data(urls, is_training=False)
        y_pred_proba = self.model.predict([X_url, X_feat], verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
        print(f"   FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")
        
        return metrics
    
    def save(self, model_dir='models'):
        """Save model"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save(model_dir / 'hybrid_model.keras')
        
        config = {
            'char_to_idx': self.char_to_idx,
            'max_url_length': self.max_url_length,
            'feature_names': self.feature_names
        }
        
        with open(model_dir / 'hybrid_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        joblib.dump(self.scaler, model_dir / 'hybrid_scaler.pkl')
        
        print(f"\n💾 Hybrid model saved to {model_dir}/")

def main():
    print("🚀 Hybrid Phishing Detection Training")
    print("="*60)
    print("Combining hand-crafted features + deep learning")
    print("="*60)
    
    # Load data
    print("\n📥 Loading datasets...")
    processed_dir = Path('data/processed')
    
    train_df = pd.read_csv(processed_dir / 'train_large.csv')
    val_df = pd.read_csv(processed_dir / 'val_large.csv')
    test_df = pd.read_csv(processed_dir / 'test_large.csv')
    
    print(f"   Training:   {len(train_df):,} URLs")
    print(f"   Validation: {len(val_df):,} URLs")
    print(f"   Test:       {len(test_df):,} URLs")
    
    # Initialize detector
    detector = HybridPhishingDetector(max_url_length=200)
    
    # Build vocabulary
    detector.build_vocabulary(train_df['url'].values)
    
    # Extract features to get feature count
    sample_features = detector.extract_features(train_df['url'].values[:100])
    num_features = sample_features.shape[1]
    
    # Build model
    detector.build_model(num_features)
    
    # Train
    history = detector.train(
        train_df['url'].values,
        train_df['label'].values,
        val_df['url'].values,
        val_df['label'].values,
        epochs=30,
        batch_size=32
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    val_metrics = detector.evaluate(val_df['url'].values, val_df['label'].values, "Validation")
    test_metrics = detector.evaluate(test_df['url'].values, test_df['label'].values, "Test")
    
    # Save
    detector.save('models')
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_type': 'Hybrid (Features + Deep Learning)',
        'training_date': datetime.now().isoformat(),
        'architecture': 'CNN + BiLSTM + Feature Dense Network',
        'num_features': num_features,
        'feature_names': detector.feature_names,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    with open(results_dir / 'hybrid_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ HYBRID TRAINING COMPLETED!")
    print("="*60)
    print(f"\nTest Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"Test F1-Score: {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()