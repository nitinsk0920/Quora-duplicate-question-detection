# Quora Duplicate Question Detection using NLP

This project focuses on detecting whether two questions asked on Quora
have the same semantic meaning, even if they are phrased differently.
The solution is implemented using Natural Language Processing (NLP) and
Machine Learning techniques.

The entire project was **implemented and executed on Kaggle using a
Kaggle-provided GPU**. The complete implementation is contained in a
single Jupyter Notebook.

---

## Problem Statement
Online Q&A platforms like Quora often contain multiple questions that
express the same intent but differ in wording. Automatically identifying
such duplicate questions helps improve search quality, reduce redundancy,
and enhance user experience.

---

## Approach
The project follows an end-to-end NLP pipeline:

1. Text preprocessing and normalization
2. Generation of semantic embeddings using **DistilBERT**
3. Feature engineering by extracting:
   - Length differences
   - Word overlap ratios
   - Fuzzy matching scores
   - Cosine similarity between embeddings
4. Training an **XGBoost classifier** on the combined feature set
5. Predicting whether a pair of questions is duplicate or not

---

## Dataset
- Quora Question Pairs Dataset
- Dataset is not included in this repository due to size constraints
- Dataset paths in the notebook are configured for the Kaggle environment

---

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Transformers (DistilBERT)
- XGBoost
- FuzzyWuzzy
- BeautifulSoup

---

## Model & Evaluation
- Model: XGBoost Classifier
- Embeddings: DistilBERT
- Validation Strategy: Cross-validation
- Accuracy Achieved: **~83%**

### Evaluation Metrics
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Execution Environment
- Platform: Kaggle Notebooks
- Hardware: Kaggle GPU
- Operating System: Linux (Kaggle Environment)

---

## How to Run the Project

### Option 1: Run on Kaggle (Recommended)
1. Upload the notebook to Kaggle
2. Add the Quora Question Pairs dataset
3. Enable GPU from **Notebook Settings**
4. Run all cells sequentially

### Option 2: Run Locally
1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
