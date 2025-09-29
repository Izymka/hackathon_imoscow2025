import shutil
import tempfile
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
import logging
import asyncio
import subprocess

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

# Импорты для ML модели
import sys
sys.path.append('../chest_ct_ai_classifier/src')

from chest_ct_ai_classifier.src.model.inference import MedicalModelInference
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
        model_path = "../chest_ct_ai_classifier/src/model/outputs/weights/best_weights.pth"
        ml_model = MedicalModelInference(model_path)
        logger.info("✅ Модель успешно загружена")
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        raise

    yield

    # Очистка ресурсов при завершении
    ml_model = None


app = FastAPI(title="DICOM Parser Service", version="0.1.0")


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


@app.get("/healthcheck")
def healthcheck() -> Dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": ml_model is not None,
        "service": "DICOM Normality Prediction",
        "time": datetime.now().isoformat()
    }


@app.post("/process")
def process(req: ProcessRequest) -> Dict[str, Any]:
    zip_path = Path(req.zip_path)

    if not zip_path.exists():
        raise HTTPException(status_code=400, detail=f"Zip path does not exist: {zip_path}")
    if not zip_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {zip_path}")
    if not _safe_is_zip(zip_path):
        raise HTTPException(status_code=400, detail=f"Path is not a valid zip archive: {zip_path}")

    tmpdir = Path(tempfile.mkdtemp(prefix="dicom_zip_"))
    extract_root = tmpdir / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_root)

        # If user specified a subdir, use it; else try to pick a sensible directory to parse
        parse_root = extract_root
        if req.extract_subdir:
            candidate = extract_root / req.extract_subdir
            if candidate.exists() and candidate.is_dir():
                parse_root = candidate
            else:
                raise HTTPException(status_code=400, detail=f"Provided extract_subdir not found: {candidate}")
        else:
            # If the archive exploded into a single top-level folder, descend into it
            children = [p for p in extract_root.iterdir() if p.is_dir()]
            if len(children) == 1:
                parse_root = children[0]

        try:
            summary: Optional[DicomSummary] = parse_dicom(parse_root, True)
            if summary is None:
                raise Exception("No DICOM files found after extraction.")
            if summary.body_part_examined != "CHEST":
                raise Exception("Expected body part examined to be CHEST, but got ", summary.body_part_examined)
        except Exception as e:
            return {
                "ok": False,
                "message": str(e),
                "extracted_to": str(extract_root),
                "parse_root": str(parse_root),
            }

        # Convert to plain dict for JSON response
        result = summary.to_dict() if hasattr(summary, 'to_dict') else dict(summary)
        return {
            "ok": True,
            "extracted_to": str(extract_root),
            "parse_root": str(parse_root),
            "summary": result,
        }

    except HTTPException:
        # Let FastAPI handle these
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        # Clean up extracted files to not consume disk
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass


def predict(dicom_dir, tensor_output_dir):
    # Вызов скрипта подготовки тензоров
    cmd = [
        "python", "../chest_ct_ai_classifier/src/scripts/prepare_ct_tensors.py",
        "--input", str(dicom_dir),
        "--output", str(tensor_output_dir)
    ]

    process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if process.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обработки DICOM: {process.stderr}"
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
    from model.inference import read_tensor
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

def cleanup_temp_dir(temp_dir: Path):
    """Фоновая задача для очистки временных файлов"""
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info(f"Временная директория очищена: {temp_dir}")
    except Exception as e:
        logger.warning(f"Не удалось очистить временную директорию {temp_dir}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8010,
        reload=True,
        log_level="debug"
    )
