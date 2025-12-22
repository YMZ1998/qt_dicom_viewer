"""RTSTRUCT parser and rasterization utilities."""
import numpy as np
import pydicom
from matplotlib.path import Path
from .utils import patient_to_index


def parse_rtstruct(path):
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
    zcount, ysize, xsize = volume_shape
    slice_masks = {}

    for roi_name, contours in rois.items():
        per_slice = {}
        for contour in contours:
            idxs = np.array([patient_to_index(pt, origin, spacing, direction) for pt in contour])
            k_vals = idxs[:, 2]
            ks = np.unique(np.round(k_vals).astype(int))
            for k in ks:
                if k < 0 or k >= zcount:
                    continue
                poly_xy = idxs[:, :2]
                mask = polygon_to_mask(poly_xy, (ysize, xsize))
                if k in per_slice:
                    per_slice[k] = per_slice[k] | mask
                else:
                    per_slice[k] = mask
        slice_masks[roi_name] = per_slice
    return slice_masks


def polygon_to_mask(poly_xy, shape):
    rows, cols = shape
    if poly_xy.shape[0] < 3:
        return np.zeros((rows, cols), dtype=bool)
    minx = int(np.floor(poly_xy[:, 0].min()))
    maxx = int(np.ceil(poly_xy[:, 0].max()))
    miny = int(np.floor(poly_xy[:, 1].min()))
    maxy = int(np.ceil(poly_xy[:, 1].max()))
    minx = max(minx, 0); maxx = min(maxx, cols - 1)
    miny = max(miny, 0); maxy = min(maxy, rows - 1)
    if minx > maxx or miny > maxy:
        return np.zeros((rows, cols), dtype=bool)
    xs = np.arange(minx, maxx + 1)
    ys = np.arange(miny, maxy + 1)
    xv, yv = np.meshgrid(xs, ys)
    points = np.vstack((xv.flatten(), yv.flatten())).T
    path = Path(poly_xy)
    mask_box = path.contains_points(points)
    mask = np.zeros((rows, cols), dtype=bool)
    mask[ys[:, None], xs[None, :]] = mask_box.reshape(len(ys), len(xs))
    return mask
