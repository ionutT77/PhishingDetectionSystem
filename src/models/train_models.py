"""
Model Training Module for Phishing Detection
Trains multiple ML models and evaluates their performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Preprocessing
from sklearn.preprocessing import StandardScaler

class PhishingDetectionTrainer:
    """Train and evaluate phishing detection models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load preprocessed feature datasets"""
        print("📥 Loading feature datasets...")
        
        features_dir = Path('data/features')
        
        # Load datasets
        train_df = pd.read_csv(features_dir / 'features_train.csv')
        val_df = pd.read_csv(features_dir / 'features_val.csv')
        test_df = pd.read_csv(features_dir / 'features_test.csv')
        
        print(f"   Training set: {len(train_df):,} samples")
        print(f"   Validation set: {len(val_df):,} samples")
        print(f"   Test set: {len(test_df):,} samples")
        
        # Separate features and labels
        X_train = train_df.drop('label', axis=1)
        y_train = train_df['label']
        
        X_val = val_df.drop('label', axis=1)
        y_val = val_df['label']
        
        X_test = test_df.drop('label', axis=1)
        y_test = test_df['label']
        
        print(f"\n📊 Features: {X_train.shape[1]} features")
        print(f"   Feature names: {list(X_train.columns)}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Standardize features"""
        print("\n🔧 Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def initialize_models(self):
        """Initialize ML models"""
        print("\n🤖 Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'Naive Bayes': GaussianNB(),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        print(f"   Initialized {len(self.models)} models")
        for name in self.models.keys():
            print(f"   ✓ {name}")
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train all models and evaluate on validation set"""
        print("\n" + "="*60)
        print("🚀 Training Models...")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\n📊 Training {name}...")
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            # Get probability predictions (for ROC-AUC)
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_val_proba = y_val_pred
                y_test_proba = y_test_pred
            
            # Calculate metrics
            metrics = {
                'training_time': training_time,
                'train': {
                    'accuracy': accuracy_score(y_train, y_train_pred),
                    'precision': precision_score(y_train, y_train_pred),
                    'recall': recall_score(y_train, y_train_pred),
                    'f1': f1_score(y_train, y_train_pred),
                },
                'validation': {
                    'accuracy': accuracy_score(y_val, y_val_pred),
                    'precision': precision_score(y_val, y_val_pred),
                    'recall': recall_score(y_val, y_val_pred),
                    'f1': f1_score(y_val, y_val_pred),
                    'roc_auc': roc_auc_score(y_val, y_val_proba),
                    'confusion_matrix': confusion_matrix(y_val, y_val_pred).tolist()
                },
                'test': {
                    'accuracy': accuracy_score(y_test, y_test_pred),
                    'precision': precision_score(y_test, y_test_pred),
                    'recall': recall_score(y_test, y_test_pred),
                    'f1': f1_score(y_test, y_test_pred),
                    'roc_auc': roc_auc_score(y_test, y_test_proba),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
                }
            }
            
            self.results[name] = metrics
            
            # Print validation results
            print(f"   ✅ Training completed in {training_time:.2f}s")
            print(f"   📈 Validation Metrics:")
            print(f"      Accuracy:  {metrics['validation']['accuracy']:.4f}")
            print(f"      Precision: {metrics['validation']['precision']:.4f}")
            print(f"      Recall:    {metrics['validation']['recall']:.4f}")
            print(f"      F1-Score:  {metrics['validation']['f1']:.4f}")
            print(f"      ROC-AUC:   {metrics['validation']['roc_auc']:.4f}")
    
    def save_results(self):
        """Save training results and models"""
        print("\n" + "="*60)
        print("💾 Saving Results...")
        print("="*60)
        
        # Create output directories
        models_dir = Path('models')
        results_dir = Path('results')
        models_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_filename = name.lower().replace(' ', '_') + '.pkl'
            model_path = models_dir / model_filename
            joblib.dump(model, model_path)
            print(f"   ✓ Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = models_dir / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"   ✓ Saved scaler to {scaler_path}")
        
        # Save results as JSON
        results_path = results_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"   ✓ Saved results to {results_path}")
        
        # Create comparison table
        self.create_comparison_table()
    
    def create_comparison_table(self):
        """Create and display model comparison table"""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON (Validation Set)")
        print("="*60)
        
        comparison_data = []
        for name, metrics in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{metrics['validation']['accuracy']:.4f}",
                'Precision': f"{metrics['validation']['precision']:.4f}",
                'Recall': f"{metrics['validation']['recall']:.4f}",
                'F1-Score': f"{metrics['validation']['f1']:.4f}",
                'ROC-AUC': f"{metrics['validation']['roc_auc']:.4f}",
                'Time(s)': f"{metrics['training_time']:.2f}"
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        print(df.to_string(index=False))
        
        # Save comparison table
        results_dir = Path('results')
        df.to_csv(results_dir / 'model_comparison.csv', index=False)
        print(f"\n✅ Comparison table saved to results/model_comparison.csv")
        
        # Find best model
        best_model = df.iloc[0]['Model']
        best_f1 = df.iloc[0]['F1-Score']
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   F1-Score: {best_f1}")
        
        # Check if meets success criteria (≥92% accuracy)
        best_accuracy = float(df.iloc[0]['Accuracy'])
        if best_accuracy >= 0.92:
            print(f"   ✅ SUCCESS! Accuracy {best_accuracy:.2%} meets ≥92% requirement")
        else:
            print(f"   ⚠️  Accuracy {best_accuracy:.2%} below 92% target")

def main():
    """Main training pipeline"""
    print("🚀 Phishing Detection Model Training")
    print("="*60)
    
    # Initialize trainer
    trainer = PhishingDetectionTrainer()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = trainer.load_data()
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
        X_train, X_val, X_test
    )
    
    # Initialize models
    trainer.initialize_models()
    
    # Train and evaluate
    trainer.train_and_evaluate(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test
    )
    
    # Save results
    trainer.save_results()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print("\n🎯 NEXT STEPS:")
    print("   1. Analyze results in results/training_results.json")
    print("   2. Create visualization notebook")
    print("   3. Start explainability module (SHAP)")
    print("   4. Build Streamlit web interface")
    print("="*60)

if __name__ == "__main__":
    main()