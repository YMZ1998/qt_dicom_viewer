"""
DICOM IO helpers: grouping series, reading series to numpy volume (with spacing/origin/direction),
and applying rescale (RescaleSlope/Intercept).
"""
import os
from collections import defaultdict
import numpy as np
import pydicom
import SimpleITK as sitk


def group_dicom_series(folder):
    """
    Scan folder and group files by SeriesInstanceUID. Returns dict{seriesuid: [filepaths]}
    """
    groups = defaultdict(list)
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                sid = getattr(ds, 'SeriesInstanceUID', None)
                if sid:
                    groups[sid].append(path)
            except Exception:
                continue
    return dict(groups)


def read_series_as_volume(file_list):
    """
    Use SimpleITK to read a series into a volume and return numpy array + meta.
    Returns: volume (z,y,x), origin (x,y,z), spacing (x,y,z), direction (3x3)
    """
    if not file_list:
        raise ValueError("Empty file_list")
    reader = sitk.ImageSeriesReader()
    # Ensure stable order: SimpleITK can sort series internally, but give file_list sorted by InstanceNumber if possible
    try:
        # try to sort by ImagePositionPatient z (if readable)
        def instance_sort_key(p):
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            if ipp:
                return float(ipp[2])
            return 0.0
        file_list_sorted = sorted(file_list, key=instance_sort_key)
    except Exception:
        file_list_sorted = sorted(file_list)
    reader.SetFileNames(file_list_sorted)
    image = reader.Execute()
    arr = sitk.GetArrayFromImage(image)  # z,y,x
    origin = np.array(image.GetOrigin())  # (x,y,z)
    spacing = np.array(image.GetSpacing())  # (x,y,z)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return arr, origin, spacing, direction


def get_rescale_from_file(path):
    """
    Read RescaleSlope/Intercept from a DICOM file (if present).
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    return slope, intercept


def apply_rescale(array, slope=1.0, intercept=0.0):
    """Apply linear rescale to convert raw pixel to HU-like values."""
    return array.astype(float) * slope + intercept
