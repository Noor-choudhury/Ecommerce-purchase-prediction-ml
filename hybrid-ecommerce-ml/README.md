# Hybrid Unsupervised-Supervised ML Architecture for E-Commerce Purchase Conversion Prediction

## 1. Executive Summary & Research Concept
Predicting e-commerce conversion intent is vital for dynamic pricing, personalized marketing, and inventory optimization. Standard pipelines pass raw tabular session logs directly into supervised models. This project introduces a **Hybrid Architecture** where unsupervised clustering (K-Means), density anomaly detection (DBSCAN), and NLP document matching (TF-IDF + Cosine Similarity) are used as **feature augmentation modules** prior to final classification via Random Forest.

## 2. Dataset Description & Scoping Bounds
- **Dataset**: UCI Online Shoppers Purchasing Intention Dataset (`12,330` sessions, `18` variables).
- **Target Variable**: `Revenue` (Binary: `0` = No Purchase, `1` = Purchase). Class imbalance: ~84.5% False, ~15.5% True.
- **Academic Disclaimer**: The original UCI dataset contains session-level aggregate metrics (e.g., `PageValues`, `BounceRates`, `ProductRelated_Duration`). It does **NOT** contain product IDs, product names, descriptions, or prices. The product catalogue and TF-IDF similarity module in this project operate on an isolated **Synthetic Demonstration Catalogue** and must not be interpreted as ground-truth user product interactions.

## 3. Methodological Framework & Anti-Leakage Controls
To guarantee strict academic validity:
1. **Stratified Split First**: A 80/20 train/test split is applied **before** fitting any scalar, encoder, K-Means clusterer, DBSCAN model, or TF-IDF vectorizer.
2. **Out-of-Sample Anomaly Scoring**: DBSCAN does not natively support `.predict()`. Test sample anomaly status is determined using nearest neighbor core distance thresholds established strictly during training.

## 4. Module Responsibilities
- `K-Means`: Captures engagement archetypes (`cluster_id`).
- `DBSCAN`: Identifies structural anomalies (`is_outlier`).
- `TF-IDF + Cosine Similarity`: Maps session profile descriptions against synthetic catalogue items (`similarity_score`).
- `Random Forest`: Takes the combined feature vector (Original UCI Features + `cluster_id` + `is_outlier` + `similarity_score`) to output calibrated purchase probabilities.

## 5. Execution Guide
### Prerequisites
```bash
pip install -r requirements.txt

python train.py


venv\Scripts\activate
streamlit run app.py