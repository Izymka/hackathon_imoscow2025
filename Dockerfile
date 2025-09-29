FROM python:3.11-slim

WORKDIR /app

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY ../chest_ct_ai_classifier/src/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Копирование проекта
COPY ../service/ /app/service/
COPY ../chest_ct_ai_classifier/src/utils /app/chest_ct_ai_classifier/src/utils
COPY ../chest_ct_ai_classifier/src/model /app/chest_ct_ai_classifier/src/model

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Открытие порта
EXPOSE 8000

# Команда для запуска в продакшене
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]