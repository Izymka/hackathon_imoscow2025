from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
import zipfile
from pathlib import Path
import tempfile
import shutil
import os

# Import the parser from the existing package
from chest_ct_ai_classifier.src.utils.dicom_parser import parse_dicom, DicomSummary

app = FastAPI(title="DICOM Parser Service", version="0.1.0")


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
    return {"status": "ok"}


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

        summary: Optional[DicomSummary] = parse_dicom(parse_root)
        if summary is None:
            return {
                "ok": False,
                "message": "No readable DICOM files found after extraction.",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )
