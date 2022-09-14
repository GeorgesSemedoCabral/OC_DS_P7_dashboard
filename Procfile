web: gunicorn code_dashboard:server

web: uvicorn code_API:app --host=0.0.0.0 --port=${PORT:-5000}