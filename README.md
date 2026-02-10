# 🛡️ ClaimWatch AI — Insurance Fraud Detection Platform

An AI-powered system that detects fraudulent insurance claims by analyzing claim details and historical patterns, and explains its reasoning in plain English.

## Tech Stack

| Layer           | Technology                                  |
|-----------------|---------------------------------------------|
| Language        | Python 3.9+                                 |
| Data            | Pandas, NumPy                               |
| ML Models       | Scikit-learn (Random Forest), XGBoost       |
| Explainability  | SHAP                                        |
| Web UI          | Flask                                       |
| Testing         | pytest                                      |

## Quick Start

```bash
# 1. Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Preprocess data
python -m src.preprocessing --category vehicle
python -m src.preprocessing --category health

# 4. Train models
python -m src.train --category vehicle
python -m src.train --category health

# 5. Launch web app
python app.py
# → Open http://127.0.0.1:5000
```

## Project Structure

```
├── app.py                  # Flask entry point
├── config.py               # Paths & hyperparameters
├── requirements.txt        # Dependencies
├── data/raw/               # Raw dataset (CSV)
├── data/processed/         # Cleaned data
├── src/
│   ├── preprocessing.py    # Module 1 — Clean & transform data
│   ├── train.py            # Module 2 — Train RF & XGBoost
│   ├── explainer.py        # Module 3 — SHAP explanations
│   └── predict.py          # Inference pipeline
├── models/                 # Saved .pkl models
├── templates/              # Flask HTML templates
├── static/                 # CSS, JS, images
└── tests/                  # pytest unit tests
```

## Running Tests

```bash
pytest tests/ -v

# Run category-specific verifications
python test_health_fraud.py
python verify_multi_insurance.py
```

## Dataset

Place your CSV dataset in `data/raw/` and update the `DATASET_FILENAME` and `TARGET_COLUMN` variables in `src/preprocessing.py`.
