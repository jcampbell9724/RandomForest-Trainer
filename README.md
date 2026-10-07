# Random Forest Trainer

A local Streamlit app that turns a CSV into an inspectable random forest model. Choose a target and features, train a classifier or regressor, compare held-out predictions with actual values, and inspect the individual trees behind the result.

## What it demonstrates

- **An end-to-end modeling workflow:** numeric imputation, categorical encoding, date expansion, train/test splitting, optional cross-validation, and model export in a scikit-learn pipeline.
- **Feature engineering with a leakage boundary:** paired columns and grouped averages; grouped averages are fitted on training rows, and custom features that depend on the target are excluded.
- **Model inspection:** feature importances, forest statistics, individual tree diagrams and rules, and manual prediction tests for saved models.
- **Traceable experiments:** selected features, parameters, split settings, and metrics are saved to a local SQLite audit history with JSONL export.
- **Accessible explanations:** parameter tooltips and an optional forest-themed story explain how the model works.

The application is a local prototype for exploration and learning. It is separate from any proprietary workplace system.

## Run locally

Use Python 3.12 in a virtual environment. From a terminal:

```powershell
git clone https://github.com/jcampbell9724/RandomForest-Trainer.git
cd RandomForest-Trainer
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Try a synthetic example

```powershell
python scripts/generate_demo.py
```

This creates `examples/synthetic_orders.csv`: 180 fictional rows with numeric, categorical, date, and missing values. No customer or employer data is included.

1. Upload the generated CSV.
2. Convert `created_at` to a date.
3. Choose `processing_days` as the target and **Regression** as the mode.
4. Use `order_value`, `item_count`, `region`, `service_tier`, and `created_at` as features. Keep the seed at 42.
5. Train, inspect the metrics and importances, download the model, then open **Inspect Saved Model** to explore its trees.

For classification, choose `service_tier` as the target and **Classification** as the mode; remove it from the feature list. Scores on this generated example demonstrate the workflow, not performance on a real business problem.

## Check the workflow

```powershell
python -m unittest discover -s tests -v
```

The synthetic checks cover classifier and regressor training, repeatable predictions, cross-validation, saved-model round trips, unseen categories, train-only grouped averages, indirect target dependencies, text target detection, and prediction on midnight or missing dates after training on timestamps.

## Practical boundaries

- Rows with a missing target are dropped. Numeric targets support regression; text, categorical, and boolean targets support classification.
- Date features become calendar and time components. Date targets are regressed on a numeric timeline and displayed as dates.
- A random holdout or ordinary cross-validation is not a time-series backtest. Choose splits that match the intended use before relying on reported scores.
- Feature importance describes this fitted model; it does not establish causation.
- Audit history is stored beside `app.py` in `training_audit.db`. Keep datasets, model exports, and audit files private when they contain sensitive information.
- Saved models use pickle, which can execute code when loaded. Open only files you created or fully trust; the viewer requires an explicit trust checkbox. Keep Python and library versions consistent between saving and loading.

## Project files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit interface, preprocessing, training, inspection, and audit storage |
| `requirements.txt` | Runtime dependencies |
| `ForestStory.docx` | Source for the optional explanation page |
| `scripts/generate_demo.py` | Deterministic synthetic CSV generator |
| `tests/test_training.py` | Focused synthetic workflow checks |

