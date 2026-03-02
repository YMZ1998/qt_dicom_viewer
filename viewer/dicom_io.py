"""DICOM IO utilities for series grouping and volume reading."""
import os
from collections import defaultdict
from functools import lru_cache
import numpy as np
import pydicom
import SimpleITK as sitk


@lru_cache(maxsize=8192)
def _read_dicom_header(path):
    return pydicom.dcmread(path, stop_before_pixels=True, force=True)


def group_dicom_series(folder):
    groups = defaultdict(list)
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                ds = _read_dicom_header(path)
                sid = getattr(ds, 'SeriesInstanceUID', None)
                if sid:
                    groups[sid].append(path)
            except Exception:
                continue
    return dict(groups)


def read_series_as_volume(file_list):
    if not file_list:
        raise ValueError("Empty file_list")
    reader = sitk.ImageSeriesReader()
    try:
        def instance_sort_key(p):
            ds = _read_dicom_header(p)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp:
                return float(ipp[2])
            return 0.0
        file_list_sorted = sorted(file_list, key=instance_sort_key)
    except Exception:
        file_list_sorted = sorted(file_list)
    reader.SetFileNames(file_list_sorted)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return arr, origin, spacing, direction


def get_rescale_from_file(path):
    ds = _read_dicom_header(path)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return slope, intercept


def apply_rescale(array, slope=1.0, intercept=0.0):
    return array.astype(float) * slope + intercept
