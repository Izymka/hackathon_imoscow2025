import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import pydicom

logging.basicConfig(level=logging.INFO)

from dataclasses import dataclass, asdict
from typing import Mapping, Iterator, Optional, Tuple

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"pydicom\.valuerep")


@dataclass(frozen=True)
class DicomSummary(Mapping[str, Any]):
    study_uid: Optional[str]
    series_uids: List[str]
    source_files_total: int
    is_multi_frame_file: bool
    n_frames: int
    rows: Optional[int]
    cols: Optional[int]
    volume_shape: Optional[Tuple[int, int, int]]
    pixel_spacing: Optional[List[float]]
    slice_thickness: Optional[float]
    spacing_between_slices: Optional[float]
    modality: Optional[str]
    rescale_slope: Optional[float]
    rescale_intercept: Optional[float]
    window_center: Optional[float]
    window_width: Optional[float]
    representative_path: Optional[str]

    # Mapping interface to preserve dict-like behavior
    def __getitem__(self, key: str) -> Any:
        return asdict(self)[key]

    def __iter__(self) -> Iterator[str]:
        return iter(asdict(self).keys())

    def __len__(self) -> int:
        return len(asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_dicom(path: str) -> bool:
    """Quickly check if a file looks like a DICOM by reading the preamble/magic.
    Falls back to trying pydicom.dcmread if needed.
    """
    try:
        with open(path, 'rb') as f:
            preamble = f.read(132)
            if len(preamble) >= 132 and preamble[128:132] == b"DICM":
                return True
    except Exception:
        return False
    # Not all DICOMs have the DICM marker; attempt to read header-only
    try:
        pydicom.dcmread(path, stop_before_pixels=True)
        return True
    except Exception:
        return False


def parse_dicom(dicom_dir: str, deep: int = 1) -> DicomSummary | None:
    """
    Parse a directory that may contain either:
      - a single multi-frame DICOM file (e.g., Enhanced CT where all frames are in one file), or
      - multiple single-frame DICOM files (one slice per file).

    Return a unified summary dict with general properties of the DICOM volume:
      {
        'study_uid': str | None,
        'series_uids': list[str],               # unique series found (most-common first)
        'source_files_total': int,              # number of readable DICOM files in directory
        'is_multi_frame_file': bool,            # True if the directory effectively represents a single multi-frame file
        'n_frames': int,                        # number of frames (slices)
        'rows': int | None,
        'cols': int | None,
        'volume_shape': tuple | None,           # (n_frames, rows, cols) when known
        'pixel_spacing': list[float] | None,    # [row_spacing_mm, col_spacing_mm]
        'slice_thickness': float | None,
        'spacing_between_slices': float | None, # estimated if possible
        'modality': str | None,
        'rescale_slope': float | None,
        'rescale_intercept': float | None,
        'window_center': float | None,
        'window_width': float | None,
        'representative_path': str | None,      # path of the representative dicom used for metadata
      }
    """
    root = Path(dicom_dir)
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Path is not a directory: {dicom_dir}")

    # 1) Collect readable DICOM files
    dicom_files: List[Path] = []

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            fp = Path(dirpath) / name
            if name.lower().endswith('.dcm') or name.lower().endswith('.dicom'):
                try:
                    pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                    dicom_files.append(fp)
                except Exception:
                    pass
            else:
                if _is_dicom(str(fp)):
                    dicom_files.append(fp)

    if not dicom_files:
        # Return empty summary
        return None

    # Read headers and gather per-file metadata
    headers = []
    processed_headers = 0
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            headers.append((fp, ds))
            processed_headers += 1
            if processed_headers >= deep:
                break
        except Exception as e:
            logging.debug(f"Skip unreadable file {fp}: {e}")

    if not headers:
        # No readable headers
        return None

    from collections import Counter

    study_uids = [getattr(ds, 'StudyInstanceUID', None) for _, ds in headers if getattr(ds, 'StudyInstanceUID', None)]
    study_uid = Counter(study_uids).most_common(1)[0][0] if study_uids else None

    series_uids_all = [getattr(ds, 'SeriesInstanceUID', None) for _, ds in headers if
                       getattr(ds, 'SeriesInstanceUID', None)]
    series_uid_ordered = [uid for uid, _ in Counter(series_uids_all).most_common()]

    # Group by SeriesInstanceUID to decide if multi-file volume or single multi-frame file
    unique_files = len(headers)

    # Heuristic: if there's exactly one file in the whole directory, assume potential multi-frame
    is_single_file = unique_files == 1

    # Choose representative header
    rep_path, rep_ds = headers[0]

    def safe_float(value):
        try:
            return float(value)
        except Exception:
            return None

    # Estimate number of frames
    n_frames: int
    is_multi_frame_file: bool

    if is_single_file:
        # Use NumberOfFrames if present, else 1
        n_frames = int(getattr(rep_ds, 'NumberOfFrames', 1) or 1)
        is_multi_frame_file = n_frames > 1
    else:
        # Pick dominant series as the volume
        dominant_series = series_uid_ordered[0] if series_uid_ordered else None
        series_headers = [(fp, ds) for fp, ds in headers if
                          getattr(ds, 'SeriesInstanceUID', None) == dominant_series] if dominant_series else headers
        # Count frames as number of files in the chosen series
        n_frames = len(series_headers)
        is_multi_frame_file = False
        # Update representative to the first file in chosen series
        rep_path, rep_ds = series_headers[0]

    rows = int(getattr(rep_ds, 'Rows', 0) or 0) or None
    cols = int(getattr(rep_ds, 'Columns', 0) or 0) or None

    pixel_spacing = None
    if hasattr(rep_ds, 'PixelSpacing'):
        try:
            pixel_spacing = [float(rep_ds.PixelSpacing[0]), float(rep_ds.PixelSpacing[1])]
        except Exception:
            try:
                # sometimes PixelSpacing is a string or other sequence
                pixel_spacing = [float(v) for v in list(rep_ds.PixelSpacing)[:2]]
            except Exception:
                pixel_spacing = None

    slice_thickness = safe_float(getattr(rep_ds, 'SliceThickness', None))

    # SpacingBetweenSlices or estimate via ImagePositionPatient z-diffs for multi-file case
    spacing_between_slices = safe_float(getattr(rep_ds, 'SpacingBetweenSlices', None))
    if spacing_between_slices is None and not is_multi_frame_file and not is_single_file:
        # try estimate from headers in chosen series
        z_positions = []
        for _, ds in headers:
            if getattr(ds, 'SeriesInstanceUID', None) != (series_uid_ordered[0] if series_uid_ordered else None):
                continue
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp is not None and len(ipp) >= 3:
                try:
                    z_positions.append(float(ipp[2]))
                except Exception:
                    pass
        if len(z_positions) >= 2:
            z_positions.sort()
            diffs = [abs(b - a) for a, b in zip(z_positions[:-1], z_positions[1:])]
            if diffs:
                # use median to be robust
                diffs.sort()
                spacing_between_slices = diffs[len(diffs) // 2]

    modality = getattr(rep_ds, 'Modality', None)

    rescale_slope = safe_float(getattr(rep_ds, 'RescaleSlope', None))
    rescale_intercept = safe_float(getattr(rep_ds, 'RescaleIntercept', None))

    # Window center/width might be sequences; take first
    def take_first_number(val):
        if val is None:
            return None
        try:
            if isinstance(val, (list, tuple)):
                return safe_float(val[0])
            return safe_float(val)
        except Exception:
            return None

    window_center = take_first_number(getattr(rep_ds, 'WindowCenter', None))
    window_width = take_first_number(getattr(rep_ds, 'WindowWidth', None))

    volume_shape = None
    if rows is not None and cols is not None and n_frames > 0:
        volume_shape = (n_frames, rows, cols)

    return DicomSummary(
        study_uid=study_uid,
        series_uids=series_uid_ordered,
        source_files_total=unique_files,
        is_multi_frame_file=is_multi_frame_file,
        n_frames=int(n_frames),
        rows=rows,
        cols=cols,
        volume_shape=volume_shape,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        spacing_between_slices=spacing_between_slices,
        modality=modality,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        window_center=window_center,
        window_width=window_width,
        representative_path=str(rep_path),
    )
