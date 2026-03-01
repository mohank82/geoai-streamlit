# GeoAI Capstone Demo (Streamlit)

## 1) Install
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate



```

## 2) Run
```bash
streamlit run app.py
```

## 3) AWS Credentials
Use any of these:
- `aws configure` (creates ~/.aws/credentials)
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` (if needed)
- IAM role (if running on EC2/SageMaker)

## 4) What to put in the sidebar
- bucket: geoai-demo-data
- state_fips: 19
- county_fips: ALL
- run_date: must match your `predictions/.../run_date=<...>/` folder
- model[season]: must match `predictions/.../model=<model_name>/`

If your prediction output is a single-column CSV, you’ll need to merge it back with the transform input order
(county/year). Best practice is to make your batch transform output include those columns.

