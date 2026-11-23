# DiaBD Diabetes Risk – Web App + API

```
diabd-app/
├── backend/
│   ├── app.py            FastAPI server (the API) – also serves the website
│   ├── train_model.py    Trains the model from the CSV -> model.joblib
│   ├── requirements.txt
│   └── data/diabd.csv    Your dataset
└── frontend/
    └── index.html        Your page, now calling the API
```

## Run it

```bash
cd backend
pip install -r requirements.txt
python train_model.py          # one time; creates model.joblib
uvicorn app:app --reload --port 8000
```

- Website:   http://localhost:8000
- API docs:  http://localhost:8000/docs   (auto-generated, try requests in the browser)

## Endpoints

| Method | URL             | Purpose                                   |
|--------|-----------------|-------------------------------------------|
| GET    | /api/health     | API status + model metrics                |
| GET    | /api/features   | Input names, units, valid ranges          |
| POST   | /api/predict    | Risk % + Low/Moderate/High + contributions |

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"age":60,"glucose":11,"hypertensive":1}'
```

Units are the model's units: **glucose in mmol/L, height in metres**
(the page converts from mg/dL and cm before sending). Any field you omit is
filled with the dataset mean.

## How the page uses it (frontend/index.html)

Search for `API integration` in the script. In short:
1. Typing updates the estimate instantly in the browser, then a debounced
   (350 ms) `POST /api/predict` replaces it with the server's answer.
2. "Save my report" fetches a fresh server prediction before saving.
3. If the API is unreachable, the page falls back to the in-browser formula
   and shows "Offline estimate".

If you host the frontend and API on different domains, change `API_BASE`
at the top of that section (CORS is already enabled in `app.py`; restrict
`allow_origins` before going live).

This is a screening estimate from a logistic regression, not a diagnosis.
