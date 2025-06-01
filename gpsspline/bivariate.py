import numpy as np
from .smooth import PySmoothingSpline

class PyBivariateSmoothingSpline:
    def __init__(self, t, x, y, k=4, lam=None, sigma=8.5, T=2, remove_velocity=True):
        self.t = t
        self.k = k
        self.T = T
        self.sigma = sigma
        self.lam = lam
        self.remove_velocity = remove_velocity

        # Fit và loại bỏ thành phần vận tốc trung bình nếu yêu cầu
        if self.remove_velocity:
            self.poly_x = np.polyfit(t, x, T + 1)
            self.poly_y = np.polyfit(t, y, T + 1)
            self.resid_x = x - np.polyval(self.poly_x, t)
            self.resid_y = y - np.polyval(self.poly_y, t)
        else:
            self.poly_x = self.poly_y = None
            self.resid_x = x
            self.resid_y = y

        self.spline_x = PySmoothingSpline(t, self.resid_x, k, lam, sigma, T)
        self.spline_y = PySmoothingSpline(t, self.resid_y, k, lam, sigma, T)

    def evaluate(self, t_eval):
        x_smooth = self.spline_x.evaluate(t_eval)
        y_smooth = self.spline_y.evaluate(t_eval)

        if self.remove_velocity:
            x_poly = np.polyval(self.poly_x, t_eval)
            y_poly = np.polyval(self.poly_y, t_eval)
            return x_smooth + x_poly, y_smooth + y_poly
        else:
            return x_smooth, y_smooth
