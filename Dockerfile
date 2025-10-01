FROM python:3.11-slim

WORKDIR /app

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Копирование и установка зависимостей (только необходимые для сервиса)
COPY service/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Копирование проекта (только нужные каталоги)
COPY service/ /app/service/
COPY chest_ct_ai_classifier/src/utils /app/chest_ct_ai_classifier/src/utils
COPY chest_ct_ai_classifier/src/model /app/chest_ct_ai_classifier/src/model
COPY chest_ct_ai_classifier/src/scripts /app/chest_ct_ai_classifier/src/scripts

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Открытие порта
EXPOSE 8000

# Команда для запуска в продакшене
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]