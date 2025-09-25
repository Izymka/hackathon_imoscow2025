import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import pydicom

logging.basicConfig(level=logging.INFO)

from dataclasses import dataclass
from typing import Mapping, Iterator, Optional, Tuple

from pydicom import config

config.enforce_valid_values = False

import warnings

logging.getLogger("pydicom").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=UserWarning, module=r"pydicom\.valuerep")
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
warnings.filterwarnings("ignore", message="Invalid value for VR.*")


@dataclass()
class DicomSummary(Mapping[str, Any]):
    # Basic study/series
    study_uid: Optional[str]
    series_uids: List[str]
    sop_class_uid: Optional[str]
    is_enhanced_ct: bool
    # Files info
    source_files_total: int
    representative_path: Optional[str]

    # Volume framing
    is_multi_frame_file: bool
    n_frames: int
    rows: Optional[int]
    cols: Optional[int]
    volume_shape: Optional[Tuple[int, int, int]]

    # Geometry
    pixel_spacing: Optional[List[float]]
    slice_thickness: Optional[float]
    spacing_between_slices: Optional[float]
    image_orientation_patient: Optional[List[float]]
    first_image_position_patient: Optional[List[float]]
    slice_location: Optional[float]

    # Patient pose
    patient_position: Optional[str]
    patient_orientation: Optional[List[str]]

    # Modality/scanner
    modality: Optional[str]
    manufacturer: Optional[str]
    manufacturer_model_name: Optional[str]
    body_part_examined: Optional[str]
    image_type: Optional[str]

    # Scan parameters
    kvp: Optional[float]
    xray_tube_current: Optional[float]
    exposure: Optional[float]
    convolution_kernel: Optional[str]
    spiral_pitch_factor: Optional[float]
    ctdi_vol: Optional[float]
    reconstruction_diameter: Optional[float]

    # Visualization / pixel data
    photometric_interpretation: Optional[str]
    bits_stored: Optional[int]
    pixel_representation: Optional[int]

    # Window & rescale (representative)
    rescale_slope: Optional[float]
    rescale_intercept: Optional[float]
    window_center: Optional[float]
    window_width: Optional[float]

    # Technical parameters
    gantry_detector_tilt: Optional[float]
    table_height: Optional[float]
    rotation_direction: Optional[str]
    focal_spots: Optional[List[float]]
    filter_type: Optional[str]
    generator_power: Optional[float]
    exposure_modulation_type: Optional[str]

    # Series-wide arrays (ordered by z)
    series_file_names: Optional[List[str]]
    instance_numbers: Optional[List[int]]
    z_positions: Optional[List[float]]
    rescale_slopes_series: Optional[List[float]]
    rescale_intercepts_series: Optional[List[float]]

    # Mapping interface to preserve dict-like behavior
    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__.keys())

    def __len__(self) -> int:
        return len(self.__dict__)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.__dict__:
            return default
        return self.__dict__[key]

    @cached_property
    def series_data_frame(self):
        names = self.get('series_file_names') or []
        insts = self.get('instance_numbers') or []
        z_positions = self.get('z_positions') or []
        slopes = self.get('rescale_slopes_series') or []
        intercepts = self.get('rescale_intercepts_series') or []

        df_z = pd.DataFrame({
            'filename': names if names else [f'frame_{i:04d}' for i in range(len(z_positions))],
            'instance_number': insts if insts else list(range(1, len(z_positions) + 1)),
            'z_position': z_positions,
            'rescale_slope': slopes if slopes else [self.rescale_slope] * len(z_positions),
            'rescale_intercept': intercepts if intercepts else [self.rescale_intercept] * len(z_positions),
        })
        return df_z.sort_values('z_position').reset_index(drop=True)

    @property
    def unique_slopes(self):
        return self.series_data_frame['rescale_slope'].dropna().unique()

    @property
    def unique_intercepts(self):
        return self.series_data_frame['rescale_intercept'].dropna().unique()

    @cached_property
    def z_clean(self):
        return self.series_data_frame['z_position'].dropna()

    @property
    def hu_volume(self):
        return ""


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


def parse_dicom(dicom_dir: str|Path) -> DicomSummary | None:
    """
    Parse a directory with CT DICOMs and return a comprehensive DicomSummary with:
      - study/series identifiers, type (enhanced vs standard), sop class uid
      - geometry and volume framing
      - scanner/scan parameters
      - window/rescale parameters
      - series-wide z-positions and rescale arrays
    """
    root = Path(dicom_dir)
    if not root.exists():
        raise ValueError(f"Path is not a directory: {dicom_dir}")

    # 1) Collect readable DICOM files
    if root.is_file() and root.suffix.lower() in ['.dcm', '.dicom']:
        dicom_files = [root]
    else:
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
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            headers.append((fp, ds))
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
        z_positions_tmp = []
        for _, ds in headers:
            if getattr(ds, 'SeriesInstanceUID', None) != (series_uid_ordered[0] if series_uid_ordered else None):
                continue
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp is not None and len(ipp) >= 3:
                try:
                    z_positions_tmp.append(float(ipp[2]))
                except Exception:
                    pass
        if len(z_positions_tmp) >= 2:
            z_positions_tmp.sort()
            diffs = [abs(b - a) for a, b in zip(z_positions_tmp[:-1], z_positions_tmp[1:])]
            if diffs:
                # use median to be robust
                diffs.sort()
                spacing_between_slices = diffs[len(diffs) // 2]

    # Basic identifiers and type
    sop_class_uid = getattr(rep_ds, 'SOPClassUID', None)
    is_enhanced_ct = bool(sop_class_uid == '1.2.840.10008.5.1.4.1.1.2.1')

    modality = getattr(rep_ds, 'Modality', None)

    # Patient pose
    patient_position = getattr(rep_ds, 'PatientPosition', None)
    patient_orientation = None
    if hasattr(rep_ds, 'PatientOrientation'):
        try:
            patient_orientation = [str(x) for x in list(rep_ds.PatientOrientation)]
        except Exception:
            try:
                patient_orientation = [str(rep_ds.PatientOrientation)]
            except Exception:
                patient_orientation = None

    # Orientation/position
    image_orientation_patient = None
    if hasattr(rep_ds, 'ImageOrientationPatient'):
        try:
            image_orientation_patient = [float(x) for x in list(rep_ds.ImageOrientationPatient)[:6]]
        except Exception:
            image_orientation_patient = None

    def get_first_ipp(ds_obj):
        ipp = getattr(ds_obj, 'ImagePositionPatient', None)
        if ipp is not None:
            try:
                return [float(ipp[0]), float(ipp[1]), float(ipp[2])]
            except Exception:
                return None
        return None

    first_image_position_patient = None
    if is_enhanced_ct and hasattr(rep_ds, 'PerFrameFunctionalGroupsSequence') and len(
            rep_ds.PerFrameFunctionalGroupsSequence) > 0:
        try:
            fg = rep_ds.PerFrameFunctionalGroupsSequence[0]
            if hasattr(fg, 'PlanePositionSequence') and len(fg.PlanePositionSequence) > 0:
                ipp = getattr(fg.PlanePositionSequence[0], 'ImagePositionPatient', None)
                if ipp is not None:
                    first_image_position_patient = [float(ipp[0]), float(ipp[1]), float(ipp[2])]
        except Exception:
            first_image_position_patient = get_first_ipp(rep_ds)
    else:
        first_image_position_patient = get_first_ipp(rep_ds)

    slice_location = safe_float(getattr(rep_ds, 'SliceLocation', None))

    # Scanner/scan parameters
    manufacturer = getattr(rep_ds, 'Manufacturer', None)
    manufacturer_model_name = getattr(rep_ds, 'ManufacturerModelName', None)
    body_part_examined = getattr(rep_ds, 'BodyPartExamined', None)
    image_type = None
    if hasattr(rep_ds, 'ImageType'):
        try:
            image_type = '\\'.join(list(rep_ds.ImageType))
        except Exception:
            image_type = str(getattr(rep_ds, 'ImageType'))

    kvp = safe_float(getattr(rep_ds, 'KVP', None))
    xray_tube_current = safe_float(getattr(rep_ds, 'XRayTubeCurrent', None))
    exposure = safe_float(getattr(rep_ds, 'Exposure', None))
    convolution_kernel = getattr(rep_ds, 'ConvolutionKernel', None)
    spiral_pitch_factor = safe_float(getattr(rep_ds, 'SpiralPitchFactor', None))
    ctdi_vol = safe_float(getattr(rep_ds, 'CTDIvol', None))
    reconstruction_diameter = safe_float(getattr(rep_ds, 'ReconstructionDiameter', None))

    # Visualization/pixel
    photometric_interpretation = getattr(rep_ds, 'PhotometricInterpretation', None)
    bits_stored = None
    try:
        v = getattr(rep_ds, 'BitsStored', None)
        bits_stored = int(v) if v is not None else None
    except Exception:
        bits_stored = None
    pixel_representation = None
    try:
        v = getattr(rep_ds, 'PixelRepresentation', None)
        pixel_representation = int(v) if v is not None else None
    except Exception:
        pixel_representation = None

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

    # Technical
    gantry_detector_tilt = safe_float(getattr(rep_ds, 'GantryDetectorTilt', None))
    table_height = safe_float(getattr(rep_ds, 'TableHeight', None))
    rotation_direction = getattr(rep_ds, 'RotationDirection', None)
    focal_spots = None
    if hasattr(rep_ds, 'FocalSpots'):
        try:
            focal_spots = [float(x) for x in list(rep_ds.FocalSpots)]
        except Exception:
            try:
                focal_spots = [float(rep_ds.FocalSpots)]
            except Exception:
                focal_spots = None
    filter_type = getattr(rep_ds, 'FilterType', None)
    generator_power = safe_float(getattr(rep_ds, 'GeneratorPower', None))
    exposure_modulation_type = getattr(rep_ds, 'ExposureModulationType', None)

    volume_shape = None
    if rows is not None and cols is not None and n_frames > 0:
        volume_shape = (n_frames, rows, cols)

    # Series-wide arrays
    series_file_names = None
    instance_numbers = None
    z_positions = None
    rescale_slopes_series = None
    rescale_intercepts_series = None

    if is_single_file and is_multi_frame_file:
        # Enhanced multi-frame
        series_file_names = [f"frame_{i:04d}" for i in range(n_frames)]
        instance_numbers = [i + 1 for i in range(n_frames)]
        z_list = []
        slopes = []
        intercepts = []
        try:
            pf_seqs = rep_ds.PerFrameFunctionalGroupsSequence
            for i in range(n_frames):
                z = None
                try:
                    fg = pf_seqs[i]
                    if hasattr(fg, 'PlanePositionSequence') and len(fg.PlanePositionSequence) > 0:
                        ipp = getattr(fg.PlanePositionSequence[0], 'ImagePositionPatient', None)
                        if ipp is not None and len(ipp) >= 3:
                            z = float(ipp[2])
                except Exception:
                    z = None
                z_list.append(z)
                slopes.append(rescale_slope if rescale_slope is not None else None)
                intercepts.append(rescale_intercept if rescale_intercept is not None else None)
        except Exception:
            pass
        z_positions = z_list
        rescale_slopes_series = slopes
        rescale_intercepts_series = intercepts
    else:
        # Standard multi-file series: gather all files from dominant series
        dominant_series = getattr(rep_ds, 'SeriesInstanceUID', None)
        if dominant_series:
            series_headers_all: List[Tuple[Path, Any]] = []
            for fp in dicom_files:
                try:
                    ds2 = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
                    if getattr(ds2, 'SeriesInstanceUID', None) == dominant_series:
                        series_headers_all.append((fp, ds2))
                except Exception:
                    continue
            # collect arrays
            names = []
            insts = []
            z_list = []
            slopes = []
            intercepts = []
            for fp, ds2 in series_headers_all:
                names.append(fp.name)
                try:
                    insts.append(int(getattr(ds2, 'InstanceNumber', 0) or 0))
                except Exception:
                    insts.append(0)
                ipp = getattr(ds2, 'ImagePositionPatient', None)
                try:
                    z_val = float(ipp[2]) if ipp is not None and len(ipp) >= 3 else None
                except Exception:
                    z_val = None
                z_list.append(z_val)
                slopes.append(safe_float(getattr(ds2, 'RescaleSlope', None)))
                intercepts.append(safe_float(getattr(ds2, 'RescaleIntercept', None)))
            # sort by z if available
            try:
                order = sorted(range(len(z_list)), key=lambda i: (float('inf') if z_list[i] is None else z_list[i]))
            except Exception:
                order = list(range(len(z_list)))
            series_file_names = [names[i] for i in order]
            instance_numbers = [insts[i] for i in order]
            z_positions = [z_list[i] for i in order]
            rescale_slopes_series = [slopes[i] for i in order]
            rescale_intercepts_series = [intercepts[i] for i in order]

    return DicomSummary(
        # Basic study/series
        study_uid=study_uid,
        series_uids=series_uid_ordered,
        sop_class_uid=sop_class_uid,
        is_enhanced_ct=is_enhanced_ct,
        # Files info
        source_files_total=unique_files,
        representative_path=str(rep_path),
        # Volume framing
        is_multi_frame_file=is_multi_frame_file,
        n_frames=int(n_frames),
        rows=rows,
        cols=cols,
        volume_shape=volume_shape,
        # Geometry
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        spacing_between_slices=spacing_between_slices,
        image_orientation_patient=image_orientation_patient,
        first_image_position_patient=first_image_position_patient,
        slice_location=slice_location,
        patient_position=patient_position,
        patient_orientation=patient_orientation,
        # Modality/scanner
        modality=modality,
        manufacturer=manufacturer,
        manufacturer_model_name=manufacturer_model_name,
        body_part_examined=body_part_examined,
        image_type=image_type,
        # Scan parameters
        kvp=kvp,
        xray_tube_current=xray_tube_current,
        exposure=exposure,
        convolution_kernel=convolution_kernel,
        spiral_pitch_factor=spiral_pitch_factor,
        ctdi_vol=ctdi_vol,
        reconstruction_diameter=reconstruction_diameter,
        # Visualization
        photometric_interpretation=photometric_interpretation,
        bits_stored=bits_stored,
        pixel_representation=pixel_representation,
        # Window & rescale
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        window_center=window_center,
        window_width=window_width,
        # Technical
        gantry_detector_tilt=gantry_detector_tilt,
        table_height=table_height,
        rotation_direction=rotation_direction,
        focal_spots=focal_spots,
        filter_type=filter_type,
        generator_power=generator_power,
        exposure_modulation_type=exposure_modulation_type,
        # Series arrays
        series_file_names=series_file_names,
        instance_numbers=instance_numbers,
        z_positions=z_positions,
        rescale_slopes_series=rescale_slopes_series,
        rescale_intercepts_series=rescale_intercepts_series,
    )
