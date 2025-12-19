"""
RTSTRUCT parser and simple rasterization utilities.

- parse_rtstruct(path): returns dict ROIName -> list of contours (each contour is Nx3 numpy array in patient coords mm)
- contours_to_slice_masks: projects contours to image index space and rasterizes per-slice masks
"""
import numpy as np
import pydicom
from matplotlib.path import Path
from .utils import patient_to_index


def parse_rtstruct(path):
    """
    Parse RTSTRUCT and return rois: dict{name -> [contour_arrays]},
    where each contour_array is (N,3) patient coords (x,y,z).
    """
    ds = pydicom.dcmread(path, force=True)
    rois = {}
    roi_name_map = {}
    for roi in getattr(ds, 'StructureSetROISequence', []) or []:
        roi_name_map[int(roi.ROINumber)] = getattr(roi, 'ROIName', f'ROI_{roi.ROINumber}')
    for rc in getattr(ds, 'ROIContourSequence', []) or []:
        rnum = int(getattr(rc, 'ReferencedROINumber', -1))
        name = roi_name_map.get(rnum, f'ROI_{rnum}')
        contours = []
        for c in getattr(rc, 'ContourSequence', []) or []:
            if not hasattr(c, 'ContourData'):
                continue
            data = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            contours.append(data)
        if contours:
            rois[name] = contours
    return rois


def contours_to_slice_masks(rois, origin, spacing, direction, volume_shape):
    """
    Convert ROI contours (patient coords) to per-slice 2D masks.
    volume_shape: (z, y, x)
    Returns dict: roi_name -> {k_slice_index: 2D bool mask (y,x)}
    Note: This implementation projects each contour's points to image index coords,
    groups by nearest integer k (z index), rasterizes polygons on that slice using matplotlib.path.Path.
    """
    zcount, ysize, xsize = volume_shape
    slice_masks = {}

    for roi_name, contours in rois.items():
        per_slice = {}
        for contour in contours:
            # project all contour points to index coords (ix, iy, iz)
            idxs = np.array([patient_to_index(pt, origin, spacing, direction) for pt in contour])  # Nx3
            # Some DICOM RT contour points may be closed (first==last). It's fine.
            # Find the integer slice indices that contour points fall into (round near)
            k_vals = idxs[:, 2]
            # We'll rasterize to the nearest k for points (tolerance 0.5)
            ks = np.unique(np.round(k_vals).astype(int))
            for k in ks:
                if k < 0 or k >= zcount:
                    continue
                # Build 2D polygon in (x, y) index space
                poly_xy = idxs[:, :2]  # (N,2): (x_index, y_index)
                # Convert to mask coordinates: rows=y (index 1), cols=x (index 0)
                mask = polygon_to_mask(poly_xy, (ysize, xsize))
                if k in per_slice:
                    per_slice[k] = per_slice[k] | mask
                else:
                    per_slice[k] = mask
        slice_masks[roi_name] = per_slice
    return slice_masks


def polygon_to_mask(poly_xy, shape):
    """
    Rasterize polygon defined by poly_xy (N,2) in index coordinates into boolean mask of shape (rows=y, cols=x).
    Uses matplotlib.path.Path for point-in-polygon test efficiently over bounding box.
    """
    rows, cols = shape
    if poly_xy.shape[0] < 3:
        return np.zeros((rows, cols), dtype=bool)
    # get bounding box in integer pixel coords
    minx = int(np.floor(poly_xy[:, 0].min()))
    maxx = int(np.ceil(poly_xy[:, 0].max()))
    miny = int(np.floor(poly_xy[:, 1].min()))
    maxy = int(np.ceil(poly_xy[:, 1].max()))
    # clip
    minx = max(minx, 0); maxx = min(maxx, cols - 1)
    miny = max(miny, 0); maxy = min(maxy, rows - 1)
    if minx > maxx or miny > maxy:
        return np.zeros((rows, cols), dtype=bool)
    # grid points in bounding box
    xs = np.arange(minx, maxx + 1)
    ys = np.arange(miny, maxy + 1)
    xv, yv = np.meshgrid(xs, ys)
    points = np.vstack((xv.flatten(), yv.flatten())).T  # (M,2)
    path = Path(poly_xy)
    mask_box = path.contains_points(points)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[ys[:, None], xs[None, :]] = mask_box.reshape(len(ys), len(xs))
    return mask
