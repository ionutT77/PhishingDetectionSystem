# 🚀 Kaggle Training Guide - 1.5M Phishing Dataset

Complete guide to upload your dataset to Kaggle and train the phishing detection models.

---

## 📋 Prerequisites

- Kaggle account (free)
- Your **UPDATED** dataset files in `data/processed/2nd_try_1mil_566k/`:
  - `train.csv` (~1.05M URLs)
  - `validation.csv` (~225K URLs)
  - `test.csv` (~225K URLs)
  - **This is the improved dataset with hard-negative clean URLs** (not the trivial one!)

---

## STEP 1️⃣: Create Kaggle Account

1. Go to https://www.kaggle.com
2. Click **"Register"** (top right)
3. Sign up with Google/email
4. Verify your email
5. ✅ **Account created!**

---

## STEP 2️⃣: Upload Dataset to Kaggle

### A. Create New Dataset

1. **Go to your profile** → Click your avatar (top right) → **"My Datasets"**

2. **Click "New Dataset"** (blue button)

3. **Upload files:**
   - Click **"Select files to upload"**
   - Upload all 3 files:
     - ✅ `train.csv`
     - ✅ `validation.csv`
     - ✅ `test.csv`
   
   **⚠️ IMPORTANT:** File upload may take 10-30 minutes for 1.5M URLs!

4. **Fill dataset information:**
   ```
   Title: Phishing URL Detection Dataset 1.5M (Hard Negatives)
   
   Subtitle: Updated dataset with complex clean URLs to prevent shortcut learning
   
   Description:
   **UPDATED DATASET** - Built specifically to avoid trivial pattern learning!
   
   - **Size:** ~1.5M URLs total (deduplicated)
   - **Balance:** 50% phishing, 50% clean URLs (perfectly balanced)
   - **Split:** 70% train, 15% validation, 15% test
   - **Quality:** Hard-negative clean URLs with parameters, deep paths, tracking codes
   - **URL Complexity:** Both phishing and clean URLs have similar length (~140-150 chars avg)
   - **Purpose:** Prevents models from learning shortcuts (e.g., "long URL = phishing")
   
   This dataset was rebuilt based on professor feedback to ensure models learn
   actual phishing patterns, not just URL length or simple structural differences.
   
   Files:
   - train.csv: Training set (~1.05M URLs)
   - validation.csv: Validation set (~225K URLs)
   - test.csv: Test set (~225K URLs)
   
   Columns:
   - url: The full URL string
   - label: 0 = clean, 1 = phishing
   ```

5. **Settings:**
   - **License:** Choose "CC0: Public Domain" or "Database: Open Database"
   - **Visibility:** 
     - **Public** (recommended) - Anyone can use it
     - **Private** - Only you can access (good for testing)

6. **Click "Create"** (blue button at bottom)

7. **Wait for processing** (may take 5-10 minutes after upload)

8. ✅ **Dataset created!** Note the dataset URL: `kaggle.com/datasets/YOUR_USERNAME/phishing-dataset-1-5m`

---

## STEP 3️⃣: Create Training Notebook

### A. Upload Notebook to Kaggle

1. **Go to Kaggle Code:** https://www.kaggle.com/code

2. **Click "+ New Notebook"** (top right)

3. **Select "Import Notebook"** → **"Upload .ipynb file"**

4. **Upload:** `helper_scripts/Training_1.5mil_dataset.ipynb`

5. **Rename notebook:** `Phishing Detection - 1.5M URLs Training`

---

### B. Configure Notebook Settings

1. **Enable GPU (FREE P100!):**
   - Right panel → **"Accelerator"** dropdown
   - Select **"GPU P100"** (⚡ 16GB VRAM, FREE!)
   
   **⚠️ If P100 not available:**
   - Try **"GPU T4 x2"** (also good)
   - Or wait a few hours and try again

2. **Enable Internet:**
   - Right panel → **"Internet"** → Toggle **ON**
   - (Needed for pip install)

3. **Add your dataset:**
   - Right panel → **"+ Add Data"**
   - Search for your dataset: `phishing-dataset-1-5m`
   - Click **"+ Add"**
   
   **⚠️ IMPORTANT:** Note the dataset path shown, e.g.:
   ```
   /kaggle/input/phishing-dataset-1-5m-hard-negatives/
   ```
   (The exact name depends on what you named it during upload)

4. **Update dataset path in notebook:**
   - Find the cell with `DATASET_PATH = '/kaggle/input/phishing-dataset/'`
   - Change to YOUR dataset name:
   ```python
   DATASET_PATH = '/kaggle/input/phishing-dataset-1-5m/'  # ← Your dataset name here!
   ```

---

### C. Run Training

1. **Click "Save Version"** (top right)
   
2. **Choose save options:**
   - ✅ **"Save & Run All"** (Recommended - runs entire notebook)
   - OR click ▶️ **"Run All"** to run without saving

3. **Wait for completion:**
   - **Expected time:** 2-4 hours for 1.5M URLs
   - **Progress tracking:**
     - 📥 Data loading: ~5-10 min
     - 🔤 Character encoding: ~15-20 min
     - 🌲 Random Forest: ~30-45 min (uses 200K subsample)
     - 🧠 Neural Network: ~90-150 min (uses full dataset)
     - 📊 Evaluation: ~10-15 min

4. **Monitor progress:**
   - Check console output for each step
   - Look for dropout messages: "High dropout (0.6-0.7) = less memorization"
   - Watch validation accuracy during NN training

---

### D. Review Results

After completion, check the notebook output:

```
✅ TRAINING COMPLETE!
=====================================
📊 Dataset size: 1,500,000 URLs
🏆 Best model: Neural Network (or Random Forest)
📈 Test Accuracy: 0.9543
📈 Test F1-Score: 0.9521
📈 Test ROC-AUC: 0.9812
=====================================
```

**✅ Success criteria:** Test Accuracy ≥ 92%

**📊 Files generated:**
- `random_forest_model.pkl` (~500 MB)
- `neural_network_model.h5` (~50 MB)
- `model_comparison.csv`
- `model_comparison.png`
- `confusion_matrices.png`

---

## STEP 4️⃣: Download Trained Models

### Option A: Direct Download (Small Files)

For files < 100 MB (e.g., `neural_network_model.h5`):

1. **In notebook output:** Look for the files in the right panel under **"Output"**
2. **Click the download icon** next to the file
3. ✅ **Downloaded!**

### Option B: Kaggle API (Large Files)

For large files like `random_forest_model.pkl`:

1. **Get Kaggle API credentials:**
   - Go to Kaggle → Your profile → **"Settings"**
   - Scroll to **"API"** section
   - Click **"Create New API Token"**
   - Downloads `kaggle.json` file

2. **On your local machine:**
   ```powershell
   # Install Kaggle CLI
   pip install kaggle
   
   # Place kaggle.json in the right location
   mkdir $env:USERPROFILE\.kaggle
   move kaggle.json $env:USERPROFILE\.kaggle\
   
   # Download notebook output
   kaggle kernels output YOUR_USERNAME/phishing-detection-1-5m-urls-training -p ./models
   ```

3. ✅ **Models downloaded to `./models/` folder!**

### Option C: Save to Kaggle Dataset Output

1. **In notebook settings:**
   - Toggle **"Save version outputs"** → ON

2. **After run completes:**
   - Go to notebook page → **"Output"** tab
   - Click **"+ New Dataset"**
   - Creates a dataset with all output files
   - Download from the new dataset page

---

## STEP 5️⃣: Verify Training Quality

### Check for Overfitting

Compare validation vs test accuracy:

```
✅ GOOD (No overfitting):
   Val Accuracy:  0.9512
   Test Accuracy: 0.9543
   Difference: 0.0031 (< 2%)

⚠️ BAD (Overfitting):
   Val Accuracy:  0.9912
   Test Accuracy: 0.8543
   Difference: 0.1369 (13.7%)
```

**If overfitting occurs:**
- High dropout (0.6-0.7) should prevent this!
- If still happens, increase dropout to 0.75 or reduce model complexity

### Check Class Balance

Look for balanced precision/recall:

```
✅ GOOD (Balanced):
              precision    recall  f1-score
Clean            0.96      0.94      0.95
Phishing         0.95      0.96      0.95

⚠️ BAD (Imbalanced):
              precision    recall  f1-score
Clean            0.99      0.75      0.85
Phishing         0.80      0.99      0.88
```

### Check Training Logs

Search for these key indicators:

```
✅ Dataset loaded correctly:
   Train: 1,050,000 (70.0%)
   Val:   225,000 (15.0%)
   Test:  225,000 (15.0%)

✅ Vocabulary built:
   Vocabulary size: 95 (should be ~80-120)

✅ High dropout active:
   ⚠️ High dropout (0.6-0.7) = less memorization = better generalization

✅ Early stopping triggered:
   Restoring model weights from the end of the best epoch: 8
```

---

## 🔧 Troubleshooting

### ❌ Problem: "File not found" error

**Solution:** Update `DATASET_PATH` in notebook to match your exact dataset name:
```python
DATASET_PATH = '/kaggle/input/YOUR-DATASET-NAME/'  # Check right panel!
```

---

### ❌ Problem: "Out of memory" error

**Solutions:**
1. **Check GPU allocation:**
   - Settings → Accelerator → Ensure **"GPU P100"** selected
   
2. **Reduce batch size** (in notebook):
   ```python
   BATCH_SIZE = 128  # Change from 256 to 128
   ```

3. **Reduce RF subsample** (if RF causes OOM):
   ```python
   SUBSAMPLE_SIZE = 100000  # Change from 200000 to 100000
   ```

---

### ❌ Problem: Training stuck at 0% for 10+ minutes

**Cause:** Large dataset loading time

**Solution:** Be patient! Initial loading takes 10-15 min for 1.5M URLs. Look for progress messages:
```
📥 Loading pre-split dataset from Kaggle...
✅ Loaded from Kaggle dataset
```

---

### ❌ Problem: Accuracy < 92%

**Possible causes:**
1. **Dataset quality:** Check if hard-negative URLs are truly complex
2. **Model underfitting:** Try reducing dropout to 0.5
3. **URL length mismatch:** Verify phishing/clean URLs have similar avg length (~140-150 chars)

**Quick fix:** Try Random Forest instead (often more robust):
```python
# Random Forest typically gets 94-96% accuracy
# Neural Network typically gets 95-97% accuracy
```

---

### ❌ Problem: "Notebook execution timeout"

**Cause:** Kaggle free tier has 9-hour limit

**Solution:**
1. **Split training into steps:**
   - Run Part 1: Data loading + RF training
   - Save RF model
   - Run Part 2: Load data + NN training

2. **Use Kaggle Pro** ($20/month):
   - 30-hour execution limit
   - Faster GPUs (A100)

---

## 📊 Expected Results

Based on 1.5M URLs with hard negatives:

| Model | Test Accuracy | Test F1 | Training Time |
|-------|--------------|---------|---------------|
| **Random Forest** | 94-96% | 0.94-0.96 | ~40 min |
| **Neural Network** | 95-97% | 0.95-0.97 | ~120 min |

**✅ Both should exceed 92% requirement!**

---

## 🎯 Next Steps

After successful training:

1. ✅ **Download models** (random_forest_model.pkl, neural_network_model.h5)

2. ✅ **Test locally:**
   ```python
   # Load model
   import joblib
   model = joblib.load('random_forest_model.pkl')
   
   # Test on new URL
   url = "http://suspicious-bank-login.tk/verify.php?id=12345"
   # ... encode URL same way as training ...
   prediction = model.predict([encoded_url])
   print("Phishing!" if prediction[0] == 1 else "Clean!")
   ```

3. ✅ **Build web interface** (Streamlit) - Ask me for help!

4. ✅ **Add explainability** (SHAP) - Show why model made decision

5. ✅ **Prepare presentation** - Show professor your results!

---

## 📚 Useful Links

- **Kaggle GPU Quotas:** https://www.kaggle.com/code
- **Kaggle Datasets:** https://www.kaggle.com/datasets
- **Kaggle API Docs:** https://github.com/Kaggle/kaggle-api
- **TensorFlow Guide:** https://www.tensorflow.org/tutorials

---

## 💡 Pro Tips

1. **Save intermediate results:**
   - Save RF model before starting NN training
   - Prevents losing 40 min of work if NN fails

2. **Use version control:**
   - Kaggle auto-saves notebook versions
   - Can revert to previous successful runs

3. **Monitor resource usage:**
   - Right panel shows GPU/RAM usage
   - If reaching limits, reduce batch size

4. **Test with subset first:**
   - Before training on 1.5M URLs, test with 100K
   - Verifies code works end-to-end
   - Takes only ~20 minutes

5. **Enable auto-save:**
   - Settings → Auto-save frequency → Every 5 minutes
   - Prevents data loss

---

**🚀 Ready to train! Good luck!**

Questions? Ask me anytime during the process! 😊
