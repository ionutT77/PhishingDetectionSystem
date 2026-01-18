"""
Model Utilities Module
Handles loading and prediction with the trained phishing detection model
"""

import numpy as np
import os
import pickle
import faulthandler


class PhishingDetector:
    def __init__(self, model_path, char_mapping_path):
        """
        Initialize the phishing detector
        
        Args:
            model_path: Path to the trained Keras model (.h5 or .keras)
            char_mapping_path: Path to the character mapping pickle file
        """
        self.model = None
        self.char_to_idx = None
        self.max_url_len = 200
        # Correct label order matching Kaggle training: 0=benign, 1=defacement, 2=malware, 3=phishing
        self.class_names = ['benign', 'defacement', 'malware', 'phishing']
        
        self.load_model(model_path)
        self.load_char_mapping(char_mapping_path)
    
    def load_model(self, path):
        """Load the trained Keras model"""
        try:
            faulthandler.enable(all_threads=True)
            print("🔧 Importing TensorFlow/Keras...", flush=True)
            from tensorflow import keras
            print("✅ TensorFlow/Keras imported", flush=True)

            # compile=False avoids deserializing optimizer/loss objects and is often more robust.
            try:
                self.model = keras.models.load_model(path, compile=False)
            except TypeError:
                self.model = keras.models.load_model(path)
            print(f"✅ Model loaded from {path}")
        except Exception as e:
            error_text = str(e)

            # Keras 3 blocks deserializing models that contain Python lambdas (Lambda layers)
            # unless unsafe deserialization is explicitly enabled.
            unsafe_deser_markers = (
                "enable_unsafe_deserialization",
                "safe_mode",
                "Lambda layer",
                "Python lambda",
                "disallowed by default",
            )

            if any(marker in error_text for marker in unsafe_deser_markers):
                try:
                    faulthandler.enable(all_threads=True)
                    print("⚠️ Retrying model load with unsafe deserialization...", flush=True)
                    from tensorflow import keras

                    if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
                        keras.config.enable_unsafe_deserialization()

                    try:
                        # safe_mode=False is required for Lambda layers saved with Keras 3.
                        self.model = keras.models.load_model(path, safe_mode=False, compile=False)
                    except TypeError:
                        # Older TF/Keras versions don't support safe_mode kwarg
                        self.model = keras.models.load_model(path, compile=False)

                    print(
                        "✅ Model loaded with unsafe deserialization enabled "
                        "(required for Lambda layers)."
                    )
                    return
                except Exception as retry_error:
                    print(f"❌ Error loading model (unsafe retry failed): {retry_error}")
                    raise

            print(f"❌ Error loading model: {e}")
            raise
    
    def load_char_mapping(self, path):
        """Load the character-to-index mapping"""
        try:
            with open(path, 'rb') as f:
                mapping = pickle.load(f)
            self.char_to_idx = mapping['char_to_idx']
            self.max_url_len = mapping['max_url_len']
            print(f"✅ Character mapping loaded from {path}")
            print(f"   Vocabulary size: {mapping['vocab_size']}")
        except Exception as e:
            print(f"❌ Error loading character mapping: {e}")
            raise
    
    def encode_url(self, url):
        """
        Encode URL to character indices (matching training preprocessing)
        
        Args:
            url: URL string to encode
            
        Returns:
            Padded sequence of character indices
        """
        # Encode characters to indices
        encoded = [self.char_to_idx.get(c, 0) for c in url[:self.max_url_len]]
        
        # Pad to max_url_len
        if len(encoded) < self.max_url_len:
            encoded += [0] * (self.max_url_len - len(encoded))
        
        return np.array([encoded])  # Shape: (1, max_url_len)
    
    def predict(self, url):
        """
        Predict the class probabilities for a given URL
        
        Args:
            url: URL string to analyze
            
        Returns:
            Dictionary containing:
                - 'probabilities': dict of class -> probability
                - 'prediction': predicted class name
                - 'confidence': confidence score (max probability)
        """
        # Encode URL to character indices
        features = self.encode_url(url)
        
        # Get predictions
        predictions = self.model.predict(features, verbose=0)[0]
        
        # Create probability dictionary
        probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(self.class_names, predictions)
        }
        
        # Get predicted class
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        return {
            'probabilities': probabilities,
            'prediction': predicted_class,
            'confidence': confidence
        }
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return "No model loaded"
        
        info = {
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_params': self.model.count_params()
        }
        
        return info
