# MSME Delinquency Risk Predictor

A Streamlit dashboard that predicts whether a funded MSME project in Western Visayas is likely
to be **Completed** or **Not Completed**, using a Ridge (L2-regularized) logistic regression model.

## Repository structure

```
.
├── app.py                      # Streamlit entrypoint (streamlit run app.py)
├── ridge_logistic_model.pkl    # Trained model bundle (model + scaler + feature order)
├── requirements.txt            # Pinned Python dependencies
├── runtime.txt                 # Python version pin
├── .streamlit/
│   └── config.toml             # Theme + server settings
└── .gitignore
```

## Model details

`ridge_logistic_model.pkl` is a pickled dict with three keys:

| Key             | Type                                  | Purpose                                              |
|-----------------|----------------------------------------|-------------------------------------------------------|
| `model`         | `sklearn.linear_model.LogisticRegressionCV` | Ridge (L2) logistic regression, tuned via 5-fold CV |
| `scaler`        | `sklearn.preprocessing.StandardScaler` | Must be applied to raw features before `predict()`   |
| `feature_names` | `list[str]`                            | Exact column order the model expects                 |

**Class encoding:** `model.classes_ == [0, 1]`, where `1 = Completed`, `0 = Not Completed`.

**Categorical encoding:** drop-first one-hot. Baseline categories (all dummies = 0) are:
`Aklan` (province), `Agriculture/Marine/Aquaculture` (sector), `Cooperative` (ownership),
`medium` (enterprise size).

`app.py` reads `feature_names` directly from the pickle at startup and validates that the
dashboard's dropdown options exactly match what the model was trained on — it will show an
error and stop rather than silently mispredict if they ever drift out of sync.

## Local setup

```bash
git clone <this-repo-url>
cd <this-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

### Streamlit Community Cloud
1. Push this repo to GitHub (public or connected private repo).
2. On [share.streamlit.io](https://share.streamlit.io), click **New app**, point it at this repo
   and `app.py`, and deploy.
3. `requirements.txt` and `runtime.txt` are picked up automatically.

### Any other host (Docker, VM, etc.)
Install `requirements.txt` into the target environment and run:
```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

## Replacing the trained model later

1. **Validate first** — load the new pickle in a separate script and confirm it has the same
   three keys (`model`, `scaler`, `feature_names`) and a sane `feature_names` list before touching
   production.
2. **Back up the current model:**
   ```bash
   cp ridge_logistic_model.pkl ridge_logistic_model_backup_$(date +%Y%m%d).pkl
   ```
3. **Replace atomically:**
   ```bash
   cp new_model.pkl ridge_logistic_model.pkl.tmp
   mv ridge_logistic_model.pkl.tmp ridge_logistic_model.pkl
   ```
4. **Restart the app** (or clear cache via Streamlit's menu during local testing) —
   `@st.cache_resource` will not pick up a changed file otherwise.
5. If the new model changes the class order, feature set, or categorical encoding, update
   `app.py` accordingly (`build_feature_vector`, the dropdown constants, and the class-order
   handling in the predict button section) — the startup check will flag a dropdown/feature
   mismatch, but it cannot detect a silent class-order flip on its own.

## Known data caveats (see analysis notebook for full detail)

- Moderate class imbalance (~63% Completed / 37% Not Completed).
- Events-per-variable is below the typical rule-of-thumb (≈6.7), which is why a regularized
  (Ridge) model was chosen over unpenalized logistic regression.
- The `Year` feature has a non-linear relationship with the outcome on the log-odds scale
  (Box-Tidwell test).
- The `Others (grouped)` sector bucket combines six distinct sub-sectors from the raw data;
  the model cannot distinguish between them.
- A small number of enterprises appear more than once in the training data (repeat funding in
  different years), so a subset of observations are not fully independent.
