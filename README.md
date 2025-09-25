# InfoLing2_project
Modular NLP pipeline for detecting rhetorical bias in news headlines. Combines emotion detection, political leaning classification, correlation analysis, and a fine-tuned DeBERTa-v3 model. Feature engineering and oversampling support performance, transparency, and reproducibility.

## Bias Detection in Political News Headlines

This repository contains a modular NLP pipeline for detecting rhetorical bias in political news headlines. It combines emotion detection, political leaning classification, correlation analysis, and a fine-tuned DeBERTa-v3 model to classify headlines as biased or unbiased.

## Project Structure

- `data/` – cleaned MBIC dataset and feature-engineered versions  
- `notebooks/` – main analysis notebook with pipeline steps  
- `models/` – saved DeBERTa model and tokenizer  
- `outputs/` – visualizations, coefficient analysis, and results  

## Pipeline Overview

1. **Emotion Detection** – using `j-hartmann/emotion-english-distilroberta-base`  
2. **Political Leaning Classification** – using `matous-volf/political-leaning-politics`  
3. **Correlation Analysis** – Pearson & Chi-Square  
4. **Politicization Analysis** – word-level and document-level  
5. **Feature Engineering** – emotion + POS features  
6. **Bias Classification** – fine-tuned DeBERTa-v3 model 

## Model Details

- Architecture: `microsoft/deberta-v3-base`  
- Task: Binary classification (`biased` vs. `unbiased`)  
- Input: Engineered emotion & POS features  
- Training: Random oversampling for class balance  
- Evaluation: Cross-validation with best fold selection

## Interpretability

Due to SHAP runtime constraints, interpretability is provided via:

- Logistic Regression Coefficient Analysis  
- Feature Importance via Random Forest (optional)

## Installation

This project relies on the following core libraries:
- Data Handling & Visualization: pandas, numpy, matplotlib, seaborn, tqdm, os – for data manipulation, visualization, and progress tracking
- NLP & Transformers: transformers, datasets, torch, huggingface_hub – for model loading, fine-tuning, and inference using DeBERTa-v3 and other transformer architectures
- Statistical Analysis: scipy.stats – for correlation and independence testing (e.g. Chi-Square)
- Evaluation & Preprocessing: scikit-learn – for metrics (accuracy, F1, precision, recall), label encoding, stratified splitting, and data preparation
- Linguistic Feature Extraction: spaCy – for part-of-speech tagging and syntactic analysis using the en_core_web_sm model
- Colab Integration & File Handling: google.colab, pickle – for saving/loading models and working with Google Drive


Clone the repository and install dependencies via:

```bash
pip install -r requirements.txt
