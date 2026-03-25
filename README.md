# Random Forest Trainer

This project is a small Streamlit app for training a random forest model from a CSV file.

## Features

- Upload a CSV file from the UI
- Convert selected columns to dates from the UI
- Choose the target column and feature columns
- Switch between auto-detect, classification, and regression modes
- Adjust common random forest parameters without editing code
- Toggle a forest-themed story page that explains how Random Forest thinks
- View train/test metrics, sample predictions, and feature importances
- Load a saved `.pkl` model from this app and inspect the forest, feature importances, and individual trees
- Download the trained model as a `.pkl` file

## Setup

```powershell
cd .\rf_public
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

## Notes

- Classification works with text or categorical targets.
- Regression requires a numeric target column.
- Date feature columns are expanded into numeric parts during training.
- Date target columns are trained through an in-app timeline-regression workaround and shown back as readable dates.
- Rows with missing target values are dropped before training.
- Training audit history is stored locally in `training_audit.db` and can be exported from the Audit Log page as JSONL when needed.
- Only load `.pkl` files you trust. Pickle files can execute code when opened.
