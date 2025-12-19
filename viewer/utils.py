"""
Utility functions for coordinate transforms between patient (mm) coords and image index coords.
"""
import numpy as np


def patient_to_index(pt_xyz, origin, spacing, direction):
    """
    Convert a point in patient coordinates (x,y,z) in mm to image index coordinates (i,j,k).
    origin: (x,y,z)
    spacing: (sx,sy,sz)
    direction: 3x3 matrix (row-major)
    Returns float indices (x_index, y_index, z_index) corresponding to image's x,y,z.
    """
    # direction is 3x3 mapping image axes to patient axes:
    # patient = direction @ (index * spacing) + origin
    dir_mat = np.array(direction).reshape(3, 3)
    relative = np.array(pt_xyz, dtype=float) - np.array(origin, dtype=float)
    inv = np.linalg.inv(dir_mat)
    idxf = inv.dot(relative) / np.array(spacing, dtype=float)
    # idxf corresponds to (i_x, i_y, i_z) with same ordering as SimpleITK (x,y,z)
    return idxf  # float indices


def indices_to_patient(index_xyz, origin, spacing, direction):
    """
    Convert index coords (i,j,k) to patient coordinates (x,y,z).
    """
    dir_mat = np.array(direction).reshape(3, 3)
    patient = dir_mat.dot(np.array(index_xyz) * np.array(spacing)) + np.array(origin)
    return patient
