import gc
import json
import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Any, Dict
import logging
import asyncio
import subprocess
import torch
import time
from pathlib import Path
import urllib.request
import os
import fcntl
import numpy as np
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Импорты для ML модели
import sys

sys.path.append('../chest_ct_ai_classifier/src')
sys.path.append('./chest_ct_ai_classifier/src')

from chest_ct_ai_classifier.src.model.prepare_tensor import prepare_ct_tensor
from chest_ct_ai_classifier.src.model.inference import MedicalModelInference
from chest_ct_ai_classifier.src.model.config import ModelConfig
from chest_ct_ai_classifier.src.utils.dicom_parser import parse_dicom, DicomSummary

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для модели
ml_model = None


def convert_numpy_types(obj: Any) -> Any:
    """
    Рекурсивно преобразует numpy и другие нестандартные типы в нативные Python типы для сериализации в JSON.
    """
    # Локальный импорт для опциональной зависимости
    try:
        from pydicom.multival import MultiValue  # type: ignore
    except Exception:  # если pydicom не установлен
        MultiValue = tuple  # подстановка, чтобы isinstance работал без pydicom

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (set, frozenset)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (datetime,)):
        return obj.isoformat()
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, MultiValue):
        # pydicom MultiValue -> список строк
        return [str(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        # Пытаемся преобразовать неизвестные объекты в строку
        try:
            return str(obj)
        except Exception:
            return obj


# Путь к скрипту подготовки данных из переменной окружения
PREPARE_CT_SCRIPT = os.getenv("PREPARE_CT_SCRIPT", "chest_ct_ai_classifier/src/scripts/prepare_ct_medicalnet_format.py")
VALID_TYPES = os.getenv("VALID_TYPES", "CHEST,CHEST_TO_PELVIS").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация модели при старте
    global ml_model
    try:
        # Загрузка весов модели
        weights_url = "https://ct-dicom-storage.website.yandexcloud.net/model/weights.pth"
        model_dir = Path("./model")
        model_path = model_dir / "weights.pth"

        # Создаём папку, если её нет
        model_dir.mkdir(parents=True, exist_ok=True)

        # Межпроцессная блокировка, чтобы не скачивать веса параллельно в нескольких воркерах
        lock_path = model_dir / "weights.lock"
        with open(lock_path, "w") as lock_file:
            logger.info("⏳ Ожидание блокировки на загрузку весов модели...")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Двойная проверка после захвата блокировки
                if not model_path.exists():
                    logger.info("📥 Загрузка весов модели...")
                    tmp_path = model_path.with_suffix(".tmp")
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)
                        start_time = time.time()
                        urllib.request.urlretrieve(weights_url, str(tmp_path))
                        # Атомарно заменяем временный файл на целевой
                        os.replace(tmp_path, model_path)
                        logger.info(f"✅ Веса успешно загружены за {time.time() - start_time:.2f} сек")
                    except Exception as download_error:
                        logger.error(f"❌ Ошибка при загрузке весов: {download_error}")
                        # Чистим временный файл на случай частичной загрузки
                        try:
                            if tmp_path.exists():
                                tmp_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        raise
                else:
                    logger.info(f"✅ Веса уже существуют: {model_path}")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        config = ModelConfig()
        ml_model = MedicalModelInference(str(model_path), config)
        logger.info("✅ Модель успешно загружена")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        raise

    yield

    # Очистка ресурсов при завершении
    ml_model = None


app = FastAPI(
    title="DICOM Parser Service",
    version="0.1.0",
    lifespan=lifespan
)


class PredictionResponse(BaseModel):
    success: bool
    patient_id: str
    prediction: Dict[str, Any]
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class ProcessRequest(BaseModel):
    zip_path: str
    extract_subdir: Optional[str] = None  # optional: if user knows exact inner dir


def _safe_is_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def read_tensor(path: Path) -> torch.Tensor:
    """Чтение тензора из PyTorch файла (.pt, .pth)"""
    try:
        tensor = torch.load(path, weights_only=False, map_location='cpu')

        # Если это словарь с 'state_dict' или 'tensor', извлекаем тензор
        if isinstance(tensor, dict):
            if 'tensor' in tensor:
                tensor = tensor['tensor']
            elif 'state_dict' in tensor:
                # Это чекпоинт модели, возвращаем как есть или извлекаем тензор
                raise ValueError(
                    "Файл содержит state_dict модели, а не тензор. Используйте load_state_dict для модели.")
            else:
                # Если словарь содержит тензор, извлекаем первый
                for key, value in tensor.items():
                    if isinstance(value, torch.Tensor):
                        return value
                raise ValueError("Не найден тензор в словаре")

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Файл не содержит тензор, а содержит: {type(tensor)}")

        return tensor

    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении PyTorch файла: {e}")


def process_predict(dicom_dir, tensor_output_dir, background_tasks: BackgroundTasks,
                    temp_dir: Path) -> PredictionResponse:
    global ml_model
    if ml_model is None:
        raise RuntimeError("Модель не загружена")

    start_time = time.time()

    try:
        tensor_result = prepare_ct_tensor(dicom_dir, tensor_output_dir, debug=True)
        logger.info("Tensors prepared", tensor_result)
    except Exception as e:
        raise HTTPException(
            status_code=509,
            detail=str(e)
        )

    # Поиск созданного тензора
    tensor_files = list(tensor_output_dir.glob("*.pt"))
    if not tensor_files:
        raise HTTPException(
            status_code=500,
            detail="Не удалось создать тензор из DICOM файлов"
        )

    tensor_file = tensor_files[0]

    # Загрузка тензора и предсказание
    tensor = read_tensor(tensor_file)

    # Предсказание
    prediction = ml_model.predict(tensor)
    # explanation = ml_model.explain_prediction(tensor, method="saliency", visualize=False, target_class=prediction["prediction"])
    del tensor
    gc.collect()

    # Преобразование numpy типов в Python типы для корректной сериализации
    prediction = convert_numpy_types(prediction)

    # Извлечение ID пациента из имени файла или метаданных
    patient_id = tensor_file.stem

    processing_time = time.time() - start_time

    # Планируем очистку временных файлов в фоне
    background_tasks.add_task(cleanup_temp_dir, temp_dir)

    return PredictionResponse(
        success=True,
        patient_id=patient_id,
        prediction=prediction,
        processing_time=processing_time,
        message="Предсказание выполнено успешно"
    )


@app.get("/healthcheck")
def healthcheck() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": ml_model is not None,
        "service": "DICOM Normality Prediction",
        "time": datetime.now().isoformat()
    }


@app.post("/process")
def process(req: ProcessRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    zip_path = Path(req.zip_path)

    logger.info(f"Processing DICOM archive: {zip_path}")

    if not zip_path.exists():
        raise HTTPException(status_code=400, detail=f"Zip path does not exist: {zip_path}")
    if not zip_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {zip_path}")
    if not _safe_is_zip(zip_path):
        raise HTTPException(status_code=400, detail=f"Path is not a valid zip archive: {zip_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="dicom_zip_"))
    extract_root = tmpdir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    tensors_dir = tmpdir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting zip archive to: {extract_root}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_root)

        # If user specified a subdir, use it; else try to pick a sensible directory to parse
        dicom_root = extract_root
        if req.extract_subdir:
            candidate = extract_root / req.extract_subdir
            if candidate.exists() and candidate.is_dir():
                dicom_root = candidate
            else:
                raise HTTPException(status_code=510, detail=f"Provided extract_subdir not found: {candidate}")
        else:
            # If the archive exploded into a single top-level folder, descend into it
            children = [p for p in extract_root.iterdir() if p.is_dir()]
            if len(children) == 1:
                dicom_root = children[0]

        try:
            summary: Optional[DicomSummary] = parse_dicom(dicom_root, True)
            if summary is None:
                raise HTTPException(511, "No DICOM files found after extraction.")
            if summary.body_part_examined not in VALID_TYPES:
                raise HTTPException(512,
                                    f"Expected body part examined to be one of {VALID_TYPES}, but got {summary.body_part_examined}")
            if summary.study_uid is None:
                raise Exception("Study UID not found")

            predict = process_predict(dicom_root, tensors_dir, background_tasks, tmpdir)

            # Convert to plain dict for JSON response
            dicom_summary = summary.to_dict() if hasattr(summary, 'to_dict') else dict(summary)

            # Сохранить результат в xlsx файл
            try:
                # Определяем путь для сохранения результата рядом с архивом
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                random_suffix = np.random.randint(1000, 9999)
                result_xlsx_path = zip_path.with_name(f"{timestamp}_{random_suffix}.xlsx")

                # Подготовка данных для сохранения
                series_uids = None
                try:
                    # summary.series_uids может быть списком/кортежем
                    series_uids = getattr(summary, "series_uids", None)
                    if isinstance(series_uids, (list, tuple)):
                        series_uids = ",".join(map(str, series_uids))
                except Exception:
                    series_uids = None

                data_row = {
                    "path_to_study": str(dicom_root),
                    "study_uid": getattr(summary, "study_uid", None),
                    "series_uid": series_uids if series_uids is not None else getattr(summary, "series_uids", None),
                    "probability_of_pathology": float(predict.prediction["probabilities"][1]) if isinstance(
                        predict.prediction.get("probabilities"), (list, tuple)) and len(
                        predict.prediction["probabilities"]) > 1 else None,
                    "pathology": int(predict.prediction.get("prediction")) if predict.prediction.get(
                        "prediction") is not None else None,
                    "processing_status": "Success",
                    "time_of_processing": float(predict.processing_time),
                }

                # Создаём DataFrame и сохраняем в .xlsx
                df = pd.DataFrame([data_row])
                df.to_excel(result_xlsx_path, index=False)
                xlsx_result_path = str(result_xlsx_path)
            except Exception as save_err:
                logger.error(f"Не удалось сохранить результат в XLSX: {save_err}")
                xlsx_result_path = None

            return {
                "ok": True,
                "extracted_to": str(extract_root),
                "dicom_root": str(dicom_root),
                "result_path": xlsx_result_path,
                "result": {
                    "message": predict.message,
                    "processing_time": predict.processing_time,
                    "prediction": {
                        "result": predict.prediction["prediction"],
                        "confidence": predict.prediction["confidence"],
                        "probabilities": list(predict.prediction["probabilities"]),
                    }
                },
                "dicom": dicom_summary,
            }

        except Exception as e:
            logger.error(f"Error processing dicom {dicom_root}: {e}")
            return {
                "ok": False,
                "message": str(e),
                "extracted_to": str(extract_root),
                "dicom_root": str(dicom_root),
            }

    except HTTPException:
        # Let FastAPI handle these
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        try:
            background_tasks.add_task(cleanup_temp_dir, tmpdir)
        except Exception:
            pass


def cleanup_temp_dir(temp_dir: Path):
    """Фоновая задача для очистки временных файлов"""
    try:
        logger.info(f"Cleanup started: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Cleanup complete: {temp_dir}")
    except Exception as e:
        logger.warning(f"Error cleaning dir {temp_dir}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="debug"
    )
