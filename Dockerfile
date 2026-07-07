FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY agents/ ./agents/
COPY skills/ ./skills/
COPY tools/ ./tools/

ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
