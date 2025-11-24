# 📘 Google Colab Usage Guide

## Quick Start

### Step 1: Upload the Notebook to Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `Phishing_Detection_Training_Colab.ipynb`

### Step 2: Upload Your Dataset

You have two options:

#### Option A: Direct Upload (Simpler, but slower for large files)
1. Run the upload cell in the notebook
2. Upload these 3 files:
   - `data/processed/train.csv` (520K URLs)
   - `data/processed/validation.csv` (65K URLs)
   - `data/processed/test.csv` (65K URLs)

#### Option B: Google Drive (Recommended for large files)
1. Upload the 3 CSV files to your Google Drive (e.g., in a folder called `PhishingDetection/data/processed/`)
2. In the notebook, uncomment the Google Drive mount code
3. Update the file paths to match your Google Drive folder structure

### Step 3: Run the Notebook

1. Click **Runtime** → **Run all** (or run cells one by one)
2. Wait for training to complete (~5-10 minutes depending on Colab resources)
3. View the results:
   - Test accuracy
   - Confusion matrix
   - Feature importance
   - Classification report

### Step 4: Download the Trained Model

The notebook will automatically download these files:
- `random_forest_model.pkl` — Trained model
- `label_encoder.pkl` — Label encoder for predictions
- `feature_columns.pkl` — Feature column names

---

## 📊 Expected Results

### Dataset Statistics
- **Total URLs**: 651,191
- **Train Set**: 520,951 URLs (80%)
- **Validation Set**: 65,120 URLs (10%)
- **Test Set**: 65,120 URLs (10%)

### URL Types Distribution
- **benign**: ~65.7% (428K URLs)
- **defacement**: ~14.8% (96K URLs)
- **phishing**: ~14.5% (94K URLs)
- **malware**: ~5.0% (32K URLs)

### Model Performance
- **Expected Accuracy**: 85-95% (depending on feature quality)
- **Training Time**: 5-10 minutes on Google Colab
- **Model Type**: Random Forest (100 trees)

---

## 🔧 Troubleshooting

### Issue: "Out of Memory" Error
**Solution**: 
- Use Google Colab Pro for more RAM
- Or reduce the dataset size by sampling

### Issue: Upload is too slow
**Solution**: 
- Use Option B (Google Drive mount)
- Files stay in Drive and load faster

### Issue: Training takes too long
**Solution**:
- Reduce `n_estimators` from 100 to 50 in the notebook
- Or reduce `max_depth` from 20 to 15

---

## 📝 How the Model Works

### URL-Based Training (No Manual Features)

The model trains **directly on the raw URL text** using **TF-IDF vectorization**:

1. **Character N-grams**: URLs are broken into 2-5 character sequences
   - Example: "https://paypal.com" → ["ht", "htt", "http", "https", "tt", "ttp", "ttps", ...]

2. **TF-IDF Weighting**: Each n-gram gets a score based on:
   - How often it appears in the URL (Term Frequency)
   - How unique it is across all URLs (Inverse Document Frequency)

3. **Feature Matrix**: Each URL becomes a vector of 5,000 numerical features

4. **Random Forest**: Learns patterns from these features to classify URLs

**Advantages:**
- ✅ No manual feature engineering
- ✅ Automatically learns URL patterns
- ✅ Captures complex character sequences
- ✅ Detects suspicious keywords and patterns

**Example Patterns Learned:**
- Phishing: "login", "verify", "account", "paypal"
- Malware: IP addresses, suspicious TLDs
- Defacement: Specific attack signatures

---

## 🎯 Using the Trained Model

After downloading the model files, you can use them to predict new URLs:

```python
import pickle

# Load model, vectorizer, and encoder
with open('url_based_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict a new URL
new_url = "http://suspicious-site.com/login"

# Vectorize the URL
url_vectorized = vectorizer.transform([new_url])

# Make prediction
prediction = model.predict(url_vectorized)[0]
predicted_label = label_encoder.inverse_transform([prediction])[0]
confidence = model.predict_proba(url_vectorized)[0][prediction]

print(f"URL Type: {predicted_label}")
print(f"Confidence: {confidence*100:.2f}%")
```

---

## 📧 Support

If you encounter any issues:
1. Check the error message in the notebook
2. Verify file paths are correct
3. Ensure all cells are run in order
4. Try restarting the runtime: **Runtime** → **Restart runtime**

---

## 🚀 Next Steps

1. **Improve the model**:
   - Add more features (domain age, SSL certificate info, etc.)
   - Try different algorithms (XGBoost, LightGBM)
   - Tune hyperparameters

2. **Deploy the model**:
   - Create a web API using Flask/FastAPI
   - Build a browser extension
   - Integrate with email filters

3. **Expand the dataset**:
   - Add more recent phishing URLs
   - Balance the classes (more malware examples)
   - Include URLs from different sources
