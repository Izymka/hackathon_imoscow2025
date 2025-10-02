FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Переменные окружения
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

#RUN pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
#    --index-url https://download.pytorch.org/whl/cpu

COPY service/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

RUN pip install --no-cache-dir scikit-learn matplotlib

# Копирование проекта (только нужные каталоги)
COPY service/ /app/service/
COPY chest_ct_ai_classifier/src/__init__.py /app/chest_ct_ai_classifier/src/__init__.py
COPY chest_ct_ai_classifier/src/utils /app/chest_ct_ai_classifier/src/utils
COPY chest_ct_ai_classifier/src/model /app/chest_ct_ai_classifier/src/model
COPY chest_ct_ai_classifier/src/scripts /app/chest_ct_ai_classifier/src/scripts

RUN apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

# Открытие порта
EXPOSE 8000

# Команда для запуска в продакшене
CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]