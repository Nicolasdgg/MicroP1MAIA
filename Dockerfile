# Dockerfile para SomnoScope - Sistema de Estadificación Polisomnográfica (MAIA)
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencias mínimas del sistema operativo
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar PyTorch optimizado para CPU (180 MB en lugar de 3 GB de CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copiar e instalar requerimientos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente, modelos y aplicaciones
COPY src/ /app/src/
COPY models/ /app/models/
COPY app/ /app/app/
COPY data/raw/sleep-cassette/SC4001* /app/data/raw/sleep-cassette/
COPY tests/ /app/tests/

EXPOSE 8000 8050 5000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
