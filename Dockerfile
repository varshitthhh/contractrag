FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install uv for fast installs
RUN pip install --no-cache-dir uv

# copy and install dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# copy application code
COPY contract_rag/ ./contract_rag/
COPY .env .env

# create log directory
RUN mkdir -p logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "contract_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]