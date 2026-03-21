# AI Student Performance Predictor + Recommendation System

This project is a full-stack ML system that:

- Uploads student CSV data
- Preprocesses and normalizes marks
- Predicts next-semester marks
- Classifies student performance and risk level
- Generates rule-based + NLP-enhanced recommendations

## Project Structure (Production-Ready)

- `app/main.py` - FastAPI entry point
- `app/routes/` - API route modules
- `app/services/` - ML/NLP/recommendation business logic
- `app/models/` - Pydantic request/response schemas
- `app/utils/` - configuration, logging, and exception handling
- `app.py` - Streamlit frontend UI
- `.env` - environment variables
- `requirements.txt` - dependencies

Top-level `main.py`, `model.py`, `nlp.py`, and `recommendation.py` are compatibility wrappers.

## Input Format

Supported file types:

- `.csv`
- `.xlsx` / `.xls`

Minimum columns:

- `Name`
- Subject columns with numeric marks (for example: `Math`, `Science`, `English`)

Optional columns:

- `attendance`
- One of: `previous_avg`, `prev_avg`, `last_sem_avg`, `prev_sem_avg`
- True next-semester targets like `next_Math`, `next_Science`, etc. (if available)

If `next_` target columns are not provided, the system creates synthetic targets so the pipeline still trains and predicts.

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `.env` to manage runtime configuration:

- `API_TITLE`
- `API_VERSION`
- `CORS_ORIGINS`
- `UPLOAD_PATH`
- `MODEL_PATH`
- `LOG_LEVEL`

## Run Tests

```bash
pytest -q
```

## One-Command Smoke Test

```bash
python smoke_test.py --csv student-scores.csv
```

This script auto-transforms the CSV schema if needed, then executes `/upload`, `/predict`, and `/recommend` in sequence.

## Run Backend (FastAPI)

```bash
uvicorn app.main:app --reload
```

Backward-compatible command still works:

```bash
uvicorn main:app --reload
```

API endpoints:

- `POST /upload` - upload CSV/Excel and train model
- `GET /predict` - get predicted marks + risk
- `GET /recommend` - get recommendations + NLP insights

## Run Frontend (Streamlit)

```bash
streamlit run app.py
```

By default, frontend calls backend at `http://127.0.0.1:8000`.

To change API URL:

```bash
set API_URL=http://127.0.0.1:8000
streamlit run app.py
```

## Run With Docker

```bash
docker compose up --build
```

- FastAPI: `http://127.0.0.1:8000/docs`
- Streamlit: `http://127.0.0.1:8501`

## Notes

- Model artifacts are saved to `artifacts/student_model.joblib`
- Latest uploaded CSV is saved to `uploads/latest_upload.csv`
- The NLP module uses `distilbert-base-uncased` with fill-mask outputs mapped to performance insights
- Logs are written to `logs/app.log`
- API docs are available at `/docs` (Swagger UI) and `/redoc`
