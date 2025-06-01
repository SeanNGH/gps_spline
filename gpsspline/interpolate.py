import numpy as np
from .bspline import PyBSpline
from scipy.interpolate import BSpline

class PyInterpolatingSpline(PyBSpline):
    def __init__(self, t, x, k):
        super().__init__(t, k)
        X = self.design_matrix(t)
        coeffs = np.linalg.solve(X, x)
        self.spline = BSpline(self.knots, coeffs, k-1)

    def __call__(self, t):
        return self.spline(t)
