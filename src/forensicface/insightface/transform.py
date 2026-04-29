# Adapted from InsightFace (MIT License): insightface/utils/transform.py
# Copyright (c) InsightFace contributors

import math
import numpy as np


def estimate_affine_matrix_3d23d(X, Y):
    """Least-squares affine matrix from 3D points X to Y."""
    X_homo = np.hstack((X, np.ones([X.shape[0], 1])))
    P = np.linalg.lstsq(X_homo, Y, rcond=None)[0].T
    return P


def P2sRt(P):
    """Decompose affine camera matrix into scale, rotation, translation."""
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def matrix2angle(R):
    """Convert rotation matrix to pitch, yaw, roll in degrees."""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz
