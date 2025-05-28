import cv2
import numpy as np

class SpatialWarp:
    """Stores and manages optimizable affine warp parameters."""
    def __init__(self):
        self.theta = 0.0 # degrees
        self.tx = 0.0    # pixels
        self.ty = 0.0    # pixels

    def get_params(self):
        return np.array([self.theta, self.tx, self.ty], dtype=np.float32)

    def update(self, params):
        # Ensure params are correctly typed before assignment if needed
        self.theta, self.tx, self.ty = float(params[0]), float(params[1]), float(params[2])


    def get_matrix(self, center_xy):
        """Calculates the 2x3 affine warp matrix."""
        rot_mat = cv2.getRotationMatrix2D(center_xy, self.theta, 1.0)
        # Combine rotation and translation
        rot_mat[0, 2] += self.tx
        rot_mat[1, 2] += self.ty
        return rot_mat

    @staticmethod
    def get_matrix_from_params(theta, tx, ty, center_xy):
        """Static method to get matrix from explicit parameters."""
        rot_mat = cv2.getRotationMatrix2D(center_xy, float(theta), 1.0) # Ensure theta is float
        rot_mat[0, 2] += float(tx) # Ensure tx is float
        rot_mat[1, 2] += float(ty) # Ensure ty is float
        return rot_mat 