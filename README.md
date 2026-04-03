# Predicting Chess Game Outcomes

A machine learning pipeline that predicts the outcome of a chess game (win, draw, or loss for the white player) using pre-game features from Chess.com's Titled Tuesday tournaments.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── chess_outcome_prediction.ipynb   # Full analysis notebook (start here)
├── scripts/
│   ├── fetch_data.py                # Step 1: Fetch raw data from Chess.com API
│   └── build_features.py            # Step 2: Feature engineering
└── data/
    ├── raw/                         # Raw API responses (JSON)
    └── dataset/                     # Processed features (CSV)
```

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

### Step 1: Fetch data from Chess.com API

Downloads tournament metadata, game records, and player statistics into `data/raw/`. Makes ~850 HTTP requests (mostly player stats) and takes a few minutes.

```bash
python scripts/fetch_data.py
```

### Step 2: Build features

Reads raw JSON files and produces train/test CSV datasets in `data/dataset/`.

```bash
python scripts/build_features.py
```

### Step 3: View the analysis

Open the notebook to see the full analysis - data exploration, model training, evaluation, and findings:

```bash
jupyter notebook chess_outcome_prediction.ipynb
```

The notebook contains:
- Feature descriptions and rationale
- Exploratory data analysis with visualizations
- Model training (Logistic Regression and Random Forest)
- Evaluation metrics and confusion matrices
- Discussion of the draw prediction challenge
- Model quality assessment and next steps

## Quick Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/build_features.py
jupyter notebook chess_outcome_prediction.ipynb
```
