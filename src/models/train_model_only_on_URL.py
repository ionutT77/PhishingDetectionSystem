"""
URL-Based Phishing Detection Model
Trains directly on raw URLs using deep learning (no feature extraction)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO/WARNING
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

class URLPhishingDetector:
    """Deep learning model that works directly on raw URLs"""
    
    def __init__(self, max_url_length=200, vocab_size=100):
        self.max_url_length = max_url_length
        self.vocab_size = vocab_size
        self.model = None
        self.char_to_idx = {}
        self.idx_to_char = {}
        
    def build_vocabulary(self, urls):
        """Build character vocabulary from URLs"""
        print("📚 Building character vocabulary...")
        
        # Get all unique characters
        all_chars = set()
        for url in urls:
            all_chars.update(url)
        
        # Create mappings (reserve 0 for padding, 1 for unknown)
        self.char_to_idx = {char: idx + 2 for idx, char in enumerate(sorted(all_chars))}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 1
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        print(f"   Vocabulary size: {len(self.char_to_idx)} characters")
        return self
    
    def encode_url(self, url):
        """Convert URL to sequence of integers"""
        # Truncate or pad URL
        url = url[:self.max_url_length]
        
        # Convert characters to indices
        encoded = [self.char_to_idx.get(char, 1) for char in url]  # 1 = <UNK>
        
        # Pad to max length
        if len(encoded) < self.max_url_length:
            encoded += [0] * (self.max_url_length - len(encoded))  # 0 = <PAD>
        
        return encoded
    
    def encode_urls(self, urls):
        """Encode multiple URLs"""
        return np.array([self.encode_url(url) for url in urls])
    
    def build_model(self):
        """Build deep learning model architecture"""
        print("\n🏗️  Building neural network architecture...")
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.max_url_length,)),
            
            # Embedding layer - learns character representations
            layers.Embedding(
                input_dim=len(self.char_to_idx),
                output_dim=64,
            ),
            
            # Convolutional layers - detect patterns in URLs
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            # Bidirectional LSTM - understand sequence context
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            
            # Dense layers - final classification
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        self.model = model
        
        print("\n📊 Model Architecture:")
        model.summary()
        
        return self
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the model"""
        print("\n🚀 Starting training...")
        print("="*60)
        
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
            )
        ]
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X, y, dataset_name="Test"):
        """Evaluate model performance"""
        print(f"\n📊 Evaluating on {dataset_name} set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        # Display results
        print(f"\n   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        cm = metrics['confusion_matrix']
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {cm[0][0]:6d}  |  FP: {cm[0][1]:6d}")
        print(f"   FN: {cm[1][0]:6d}  |  TP: {cm[1][1]:6d}")
        
        return metrics
    
    def predict_url(self, url):
        """Predict if a single URL is phishing"""
        encoded = self.encode_url(url)
        encoded = np.array([encoded])
        
        proba = self.model.predict(encoded, verbose=0)[0][0]
        prediction = 1 if proba >= 0.5 else 0
        
        return {
            'url': url,
            'prediction': 'PHISHING' if prediction == 1 else 'LEGITIMATE',
            'confidence': float(proba) if prediction == 1 else float(1 - proba)
        }
    
    def save(self, model_dir='models'):
        """Save model and vocabulary"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(model_dir / 'url_phishing_model.keras')
        
        # Save vocabulary and config
        config = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'max_url_length': self.max_url_length,
            'vocab_size': self.vocab_size
        }
        
        with open(model_dir / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Model saved to {model_dir}/")
    
    def load(self, model_dir='models'):
        """Load saved model"""
        model_dir = Path(model_dir)
        
        # Load Keras model
        self.model = keras.models.load_model(model_dir / 'url_phishing_model.keras')
        
        # Load vocabulary and config
        with open(model_dir / 'model_config.json', 'r') as f:
            config = json.load(f)
        
        self.char_to_idx = config['char_to_idx']
        self.idx_to_char = config['idx_to_char']
        self.max_url_length = config['max_url_length']
        self.vocab_size = config['vocab_size']
        
        print(f"✅ Model loaded from {model_dir}/")
        return self


def main():
    """Main training pipeline"""
    print("🚀 URL-Based Phishing Detection Training")
    print("="*60)
    print("Training directly on raw URLs (no feature extraction)")
    print("="*60)
    
    # Load data 

    print("\n📥 Loading preprocessed datasets...")
    processed_dir = Path('data/processed')
    
    train_df = pd.read_csv(processed_dir / 'train_large.csv')
    val_df = pd.read_csv(processed_dir / 'val_large.csv')
    test_df = pd.read_csv(processed_dir / 'test_large.csv')
    
    print(f"   Training set:   {len(train_df):,} URLs")
    print(f"   Validation set: {len(val_df):,} URLs")
    print(f"   Test set:       {len(test_df):,} URLs")
    
    # Initialize detector
    detector = URLPhishingDetector(max_url_length=200)
    
    # Build vocabulary from training URLs
    detector.build_vocabulary(train_df['url'].values)
    
    # Encode URLs
    print("\n🔤 Encoding URLs to numerical sequences...")
    X_train = detector.encode_urls(train_df['url'].values)
    y_train = train_df['label'].values
    
    X_val = detector.encode_urls(val_df['url'].values)
    y_val = val_df['label'].values
    
    X_test = detector.encode_urls(test_df['url'].values)
    y_test = test_df['label'].values
    
    print(f"   Training shape: {X_train.shape}")
    print(f"   Each URL encoded as {detector.max_url_length} integers")
    
    # Build model
    detector.build_model()
    
    # Train model
    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=20,
        batch_size=32
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    val_metrics = detector.evaluate(X_val, y_val, "Validation")
    test_metrics = detector.evaluate(X_test, y_test, "Test")
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_type': 'Deep Learning (CNN + BiLSTM)',
        'training_date': datetime.now().isoformat(),
        'architecture': 'Embedding + Conv1D + BiLSTM',
        'max_url_length': detector.max_url_length,
        'vocabulary_size': len(detector.char_to_idx),
        'validation': val_metrics,
        'test': test_metrics
    }
    
    with open(results_dir / 'url_model_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir}/url_model_results.json")
    
    # Save model
    detector.save('models')
    
    # Test predictions on sample URLs
    print("\n" + "="*60)
    print("🧪 TESTING ON SAMPLE URLs")
    print("="*60)
    
    test_urls = [
        test_df[test_df['label'] == 0]['url'].iloc[0],  # Legitimate
        test_df[test_df['label'] == 1]['url'].iloc[0],  # Phishing
    ]
    
    for url in test_urls:
        result = detector.predict_url(url)
        print(f"\nURL: {url[:80]}...")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print("\n🎯 SUCCESS CRITERIA CHECK (≥92% Accuracy):")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.2%}", end="")
    if test_metrics['accuracy'] >= 0.92:
        print(" ✅ PASS (≥92%)")
    else:
        print(" ❌ FAIL (<92%)")
    
    print("\n🎯 NEXT STEPS:")
    print("   1. Model saved in models/url_phishing_model.keras")
    print("   2. Test with: python src/models/predict_url.py <url>")
    print("   3. Build Streamlit interface")
    print("="*60)


if __name__ == "__main__":
    main()