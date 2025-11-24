# 🔒 Phishing Detection System

A machine learning-based system for automatically detecting malicious URLs with explainability and confidence scoring.

## 🎯 Project Overview

This project builds an intelligent system that can automatically detect whether a URL is:
- **Benign** - Legitimate, safe URLs
- **Defacement** - Compromised or altered websites
- **Phishing** - Malicious URLs attempting to steal information
- **Malware** - URLs hosting malicious software

The system leverages supervised machine learning and includes:
- ✨ **Explainability Module** - Shows why a URL was flagged as suspicious
- 📊 **Confidence Scoring** - Provides probability scores for each prediction
- 🖥️ **Interactive Demo Interface** - User-friendly demonstration tool

## 🎓 Project Goals

### Success Criteria
- **Accuracy**: ≥ 92% or F1-score ≥ 0.85 for "phishing" class
- **False Positive Rate**: < 8%
- **Explainability**: Available for at least 95% of predictions

## 📋 Project Phases

### 1. Project Definition & Setup
- Define objectives and success criteria
- Identify data sources and evaluation plan
- Establish project scope

### 2. Data Collection & Preparation
**Data Sources:**
- **Phishing URLs**: PhishTank, OpenPhish, Kaggle datasets
- **Clean URLs**: Tranco Top 1M, Alexa lists
- **Recent Data**: 2024-2025 URLs for current phishing patterns

**Key Tasks:**
- Collect diverse URL datasets
- Manual validation of subset (1,000-3,000 URLs)
- Store URLs with metadata (source, WHOIS age, domain, label) in structured format (CSV/JSON)

### 3. Feature Extraction
Transform URLs into numerical and categorical features:

**Lexical Features:**
- URL length
- Special characters count
- Number of tokens
- Entropy

**Host-Based Features:**
- TLD (Top-Level Domain) type
- Domain age (WHOIS)
- IP address usage instead of domain name

**Heuristic Features:**
- Brand name similarity
- Suspicious tokens
- URL shortener detection

**Optional Content Features:**
- Page title
- Redirections
- Login forms presence

### 4. Model Design & Training
**Approach:**
- Start with classic ML models: Logistic Regression, Random Forest, XGBoost/LightGBM
- Handle class imbalance
- Hyperparameter optimization
- Model calibration for realistic confidence scores
- Save trained model and preprocessing pipeline

### 5. Explainability & Confidence Scoring
**Implementation:**
- Feature importance analysis
- SHAP values for individual predictions
- Generate human-readable explanations:
  - "Domain mimics paypal.com"
  - "WHOIS age < 30 days"
- Calibrate probabilities for accurate confidence scores

**Output:** Top 3 reasons + confidence scores for each prediction

### 6. Evaluation & Manual Validation
**Metrics:**
- Accuracy, Precision, Recall, F1-score (especially for phishing class)
- Expected Calibration Error (ECE) for confidence reliability
- Temporal validation on recent data (2024-2025)
- Final evaluation on manually validated test set

### 7. User Interface & Demo
**Features:**
- Text/URL input field
- Output display: predicted label, confidence score, top 3 explanatory factors
- Optional visualizations: redirections, domain age
- Batch processing capability (CSV upload)

**Technology:** Streamlit or similar framework

### 8. Final Reporting & Presentation
**Deliverables:**
- Implementation report
- Live demonstration
- Presentation slides (context, methodology, results)
- Well-documented code repository

## 📅 Timeline (10 Weeks)

| Week | Activity |
|------|----------|
| 0 | Finalize proposal with supervisor |
| 1-2 | Data collection (historical + 2024-2025 URLs) |
| 3 | Implement feature extraction pipeline |
| 4 | Baseline models (Logistic Regression / Random Forest) |
| 5 | Model calibration + SHAP explainability integration |
| 6 | Advanced models (XGBoost / LightGBM) |
| 7 | Demo interface development |
| 8 | Manual validation + final evaluation |
| 9 | Report writing and presentation preparation |
| 10 | Final delivery and project defense |

## 🔍 Key Concepts

### WHOIS Age
WHOIS is a protocol and online service that provides public information about domain names (e.g., google.com) or IP addresses.

**Why it matters for phishing detection:**
- **Phishing domains** are typically newly created (e.g., 5 days old)
- **Legitimate domains** are old, often thousands of days old (e.g., paypal.com, amazon.com)
- **Low WHOIS age** is a risk signal that the URL might be phishing

## 🚀 Getting Started

### Option 1: Train on Google Colab (Recommended)

1. **Upload the notebook to Google Colab**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Upload `Phishing_Detection_Training_Colab.ipynb`

2. **Upload your dataset**
   - Upload `data/processed/train.csv`, `validation.csv`, and `test.csv`
   - Or mount Google Drive if files are large

3. **Run the notebook**
   - Click **Runtime** → **Run all**
   - Wait ~5-10 minutes for training

4. **Download the trained model**
   - Model files will be automatically downloaded

📖 **See [COLAB_USAGE.md](COLAB_USAGE.md) for detailed instructions**

---

### Option 2: Train Locally

```bash
# Clone the repository
git clone https://github.com/ionutT77/PhisingDetectionSystem.git

# Navigate to project directory
cd PhisingDetectionSystem

# Install dependencies
pip install -r requirements.txt

# Split the dataset
python src/data_processing/train_test_split.py

# Train the model
python train_model_colab.py
```

## 📊 Dataset Statistics

- **Total URLs**: 651,191
- **Train Set**: 520,951 URLs (80%)
- **Validation Set**: 65,120 URLs (10%)
- **Test Set**: 65,120 URLs (10%)

**URL Type Distribution**:
- Benign: ~428,103 URLs (65.7%)
- Defacement: ~96,457 URLs (14.8%)
- Phishing: ~94,111 URLs (14.5%)
- Malware: ~32,520 URLs (5.0%)

## 📊 Expected Results

- High-accuracy phishing detection with explainable predictions
- Reliable confidence scores for decision-making
- User-friendly interface for testing and demonstration
- Comprehensive evaluation report with error analysis

## 🛠️ Tech Stack

- **Python** - Core programming language
- **Scikit-learn** - Machine learning models
- **XGBoost/LightGBM** - Advanced gradient boosting
- **SHAP** - Model explainability
- **Streamlit** - Web interface
- **Pandas/NumPy** - Data manipulation

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

## 📧 Contact

For questions or feedback, please contact [your contact information]
