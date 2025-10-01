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

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Импорты для ML модели
import sys

sys.path.append('../chest_ct_ai_classifier/src')

from chest_ct_ai_classifier.src.model.inference import MedicalModelInference
from chest_ct_ai_classifier.src.model.config import ModelConfig
from chest_ct_ai_classifier.src.utils.dicom_parser import parse_dicom, DicomSummary

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальная переменная для модели
ml_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Инициализация модели при старте
    global ml_model
    try:
        config = ModelConfig()
        model_path = "../chest_ct_ai_classifier/src/model/outputs/weights/best-epoch=38-val_f1=0.7917-val_auroc=0.8628.ckpt"
        ml_model = MedicalModelInference(model_path, config)
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


def process_predict(dicom_dir, tensor_output_dir, background_tasks: BackgroundTasks, temp_dir: Path) -> PredictionResponse:

    global ml_model
    if ml_model is None:
        raise RuntimeError("Модель не загружена")

    start_time = time.time()

    # Вызов скрипта подготовки тензоров
    cmd = [
        "python", "../chest_ct_ai_classifier/src/scripts/prepare_ct_medicalnet_format.py",
        "--input", str(dicom_dir),
        "--output", str(tensor_output_dir)
    ]

    process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки DICOM: {process.stderr}"
        )

    logger.info("Tensors prepared", extra={"stdout": process.stdout, "stderr": process.stderr})

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
                raise HTTPException(status_code=400, detail=f"Provided extract_subdir not found: {candidate}")
        else:
            # If the archive exploded into a single top-level folder, descend into it
            children = [p for p in extract_root.iterdir() if p.is_dir()]
            if len(children) == 1:
                dicom_root = children[0]

        try:
            summary: Optional[DicomSummary] = parse_dicom(dicom_root, True)
            if summary is None:
                raise Exception("No DICOM files found after extraction.")
            if summary.body_part_examined != "CHEST":
                raise Exception("Expected body part examined to be CHEST, but got ", summary.body_part_examined)
            if summary.study_uid is None:
                raise Exception("Study UID not found")

            predict = process_predict(dicom_root, tensors_dir, background_tasks, tmpdir)

            # Convert to plain dict for JSON response
            result = summary.to_dict() if hasattr(summary, 'to_dict') else dict(summary)
            return {
                "ok": True,
                "extracted_to": str(extract_root),
                "dicom_root": str(dicom_root),
                "summary": result,
                "result": predict
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
