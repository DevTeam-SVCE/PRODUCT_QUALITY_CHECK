"""Drawing mm -> image px mappings: flat similarity (Sim), perspective (Proj), and the weighted fit."""
import math

import cv2
import numpy as np


class Sim:
    """x_img = s * R(theta) * M * x_dwg + t, M = diag(1, -1) when the photo shows the mirror image."""

    def __init__(self, s, theta, t, mirror):
        self.s, self.theta, self.t, self.mirror = s, theta, np.asarray(t, float), mirror

    @property
    def A(self):
        c, sn = math.cos(self.theta), math.sin(self.theta)
        M = np.diag([1.0, -1.0]) if self.mirror else np.eye(2)
        return self.s * np.array([[c, -sn], [sn, c]]) @ M

    def __call__(self, p):
        return np.asarray(p, float) @ self.A.T + self.t

    def inverse(self, p):
        return (np.asarray(p, float) - self.t) @ np.linalg.inv(self.A).T

    def local_scale(self, x, y):
        return self.s


class Proj:
    """Perspective (homography) mapping drawing mm -> image px, for a flat face photographed at an angle.
    Offers the same interface as Sim; s/A/t/theta describe the local linear behaviour at `ref` (mm)."""

    def __init__(self, H, ref):
        self.H = np.asarray(H, float)
        self.Hi = np.linalg.inv(self.H)
        self.ref = np.asarray(ref, float)
        J = self.jacobian(*self.ref)
        self.mirror = bool(np.linalg.det(J) < 0)
        self.s = math.sqrt(abs(np.linalg.det(J)))
        self.A = J
        self.t = self(self.ref[None])[0] - J @ self.ref
        B = J @ np.diag([1.0, -1.0]) if self.mirror else J
        self.theta = math.atan2(B[1, 0], B[0, 0])

    def __call__(self, p):
        p = np.asarray(p, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.H).reshape(-1, 2)

    def inverse(self, p):
        p = np.asarray(p, float).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.Hi).reshape(-1, 2)

    def jacobian(self, x, y, h=0.01):
        c = self(np.array([[x, y], [x + h, y], [x, y + h]]))
        return np.c_[(c[1] - c[0]) / h, (c[2] - c[0]) / h]

    def local_scale(self, x, y):
        return math.sqrt(abs(np.linalg.det(self.jacobian(x, y))))


def fit_similarity(src, dst, mirror, w=None, fixed_scale=None):
    """Weighted Umeyama similarity (no reflection beyond the fixed mirror)."""
    src = np.asarray(src, float) @ (np.diag([1.0, -1.0]) if mirror else np.eye(2)).T
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    W = w / w.sum()
    ms, md = (W[:, None] * src).sum(0), (W[:, None] * dst).sum(0)
    S, D = src - ms, dst - md
    cov = (W[:, None] * D).T @ S
    U, sig, Vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ np.diag([1.0, d]) @ Vt
    s = fixed_scale or float((sig * np.array([1.0, d])).sum() / (W * (S ** 2).sum(1)).sum())
    t = md - s * R @ ms
    return Sim(s, math.atan2(R[1, 0], R[0, 0]), t, mirror)
