# InfoLing2_project
Modular NLP pipeline for detecting rhetorical bias in news headlines. Combines emotion and ideology features with a fine-tuned DeBERTa-v3 model. Stratified K-Fold validation ensures robust, reproducible performance across text-only and feature-enhanced setups.

## Project Structure

- `data/` – combined dataset of MBIC and BEADs with additional features "emotion scores", "dominant emotion", "leaning scores" and "political leaning"
- `notebook/` – main analysis notebook with pipeline steps (Note: Due to widget metadata, GitHub may not render the notebook preview correctly. Please open it locally for full functionality.)
- `results/`: csv-files of all preditions and all fold metrics from both model trainings

## Datasets
- MBIC (A Media Bias Annotation Dataset) sourced from: https://www.kaggle.com/datasets/timospinde/mbic-a-media-bias-annotation-dataset?resource=download
- BEADs (Bias Evaluation Across Domains Dataset) sourced from: https://huggingface.co/datasets/shainar/BEAD

## Pipeline Overview

2. **Baseline text-only classifier** with a DeBERTa-v3 model (https://huggingface.co/microsoft/deberta-v3-base)
3. **Emotion Detection** using `j-hartmann/emotion-english-distilroberta-base` (https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
4. **Political Leaning Classification** using `matous-volf/political-leaning-politics` (https://huggingface.co/matous-volf/political-leaning-politics?text=Opting+for+abortion+is+an+inalienable+right+of+every+individual.) 
5. **Politicization Analysis** – word-level and document-level  
6. **Correlation Analysis** – Pearson & Chi-Square  
7. **Feature Engineering** – emotion + POS features  
8. **Feature-enhanced classifer** with a DeBERTa-v3 model and fushion architecture

## Model Details

- Architecture: `microsoft/deberta-v3-base`
- Task: label classification (`left`, `right`, `center` vs. `unbiased`)  
- Input: Text-only (headlines) and additional features (dominant emotion, political leaning prediction & politicization score)  
- Training: Random oversampling for class balance  
- Evaluation: Cross-validation with best fold selection

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
