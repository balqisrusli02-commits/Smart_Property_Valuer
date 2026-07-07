# Smart Property Valuer (Flask + XGBoost + MySQL)

Proposal-aligned FYP web app with:
- Property price prediction (XGBoost model)
- Property recommendation (cosine similarity)
- Property search/filter page
- Property detail page
- Nearest hospital/primary school/secondary school distance
- Prediction history in MySQL

## 1) Prerequisites

- XAMPP (MySQL started)
- Python 3.10+
- Trained files:
  - `best_model.pkl`
  - `model_columns.pkl`

## 2) Setup

1. Open terminal in:
   - `c:\xampp\htdocs\Smart Property Valuer FYP`
2. Create environment and install:
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
   - `pip install -r requirements.txt`
3. Copy env file:
   - `copy .env.example .env`
4. Place model artifacts:
   - `models/best_model.pkl`
   - `models/model_columns.pkl`
5. Set dataset paths in `.env` (property, property-src, hospital, primary school, secondary school, property image directory).
6. (Optional, for dynamic LLM suggestions) set `LLM_API_KEY`, `LLM_API_URL`, and `LLM_MODEL` in `.env`.
7. Ensure MySQL DB exists:
   - Import `schema.sql` in phpMyAdmin, or
   - Run app once and open `/init-db` (auto-create tables)

## 3) Run

- `python app.py`
- Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 4) Main Pages

- `/` - Home Page
- `/predict` - Price Prediction + Property Recommendation
- `/search` - Property Search & Filters
- `/property/<id>` - Property Details
- `/history` - Prediction History
- `/import-property-data` - Reload property table from CSV dataset

## 5) Notes

- Feature order is loaded automatically from `model_columns.pkl`.
- Categorical codes follow your notebook mapping (Furnishing, Tenure, Property_Type, Unit_Type).
- If prediction errors happen, check model file path and feature column compatibility.
- Distances are computed with Haversine formula from property coordinates to nearest facilities.
- Each nearest facility card includes a Google Maps button.
- Search page supports API-based LLM suggestions (with safe rule-based fallback when API is not configured).
- Property detail page image fallback order:
  1) local file by ID pattern (example: `P1224.jpg`) from `PROPERTY_IMAGE_DIR`
  2) URL from `src` column in `PROPERTY_SRC_DATASET_PATH` (or `PROPERTY_DATASET_PATH`) matched by `Property_ID`
  3) map fallback using property coordinates
