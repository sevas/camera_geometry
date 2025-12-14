import numpy as np
import cv2


def project_points_cv(points, k, dist_coeffs):
    tvec = np.array([0.0, 0.0, 0.0])
    rvec = np.eye(3)
    cv_projected, _ = cv2.projectPoints(
        points, cameraMatrix=k, distCoeffs=dist_coeffs, rvec=rvec, tvec=tvec
    )
    uv = np.squeeze(cv_projected)
    return uv
