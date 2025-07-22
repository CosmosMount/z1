import math
import numpy as np

from utils import grd_yup2grd_zup, T_base_to_link06, mat_update, fast_mat_inv

class VuerPreprocessor:
    def __init__(self):
        self.initialized = False
        self.connected = False
        self.vuer_right_controller_mat = np.eye(4)
        self.vuer_left_controller_mat = np.eye(4)
        self.T_align_right = np.eye(4)
    def process(self, tracker):
        self.vuer_right_controller_mat = mat_update(self.vuer_right_controller_mat, tracker.right_controller.copy())
        self.vuer_left_controller_mat = mat_update(self.vuer_left_controller_mat, tracker.left_controller.copy())

        self.vuer_right_controller_mat = grd_yup2grd_zup @ self.vuer_right_controller_mat @ fast_mat_inv(grd_yup2grd_zup)
        self.vuer_left_controller_mat = grd_yup2grd_zup @ self.vuer_left_controller_mat @ fast_mat_inv(grd_yup2grd_zup)

        if self.connected:
            if not self.initialized:
                self.T_align_right = T_base_to_link06 @ fast_mat_inv(self.vuer_right_controller_mat)
                self.initialized = True

        rel_right_controller_mat = self.T_align_right @ self.vuer_right_controller_mat
        rel_left_controller_mat = self.vuer_left_controller_mat

        return rel_right_controller_mat, rel_left_controller_mat
    
    