"""Coordinate transformation utilities between patient and image index space."""
import numpy as np


def patient_to_index(pt_xyz, origin, spacing, direction):
    dir_mat = np.array(direction).reshape(3, 3)
    relative = np.array(pt_xyz, dtype=float) - np.array(origin, dtype=float)
    inv = np.linalg.inv(dir_mat)
    idxf = inv.dot(relative) / np.array(spacing, dtype=float)
    return idxf


def patient_points_to_indices(points_xyz, origin, spacing, direction):
    """Vectorized variant of patient_to_index for an (N, 3) points array."""
    dir_mat = np.array(direction, dtype=float).reshape(3, 3)
    points = np.asarray(points_xyz, dtype=float)
    relative = points - np.asarray(origin, dtype=float)
    inv = np.linalg.inv(dir_mat)
    return (relative @ inv.T) / np.asarray(spacing, dtype=float)


def indices_to_patient(index_xyz, origin, spacing, direction):
    dir_mat = np.array(direction).reshape(3, 3)
    patient = dir_mat.dot(np.array(index_xyz) * np.array(spacing)) + np.array(origin)
    return patient
