import numpy as np
from scipy.interpolate import BSpline
from .knots import not_a_knot_vector

class PyBSpline:
    def __init__(self, t, k, knot_method="not_a_knot"):
        self.t = np.asarray(t)
        self.k = k
        self.n = len(t)

        # Chọn phương pháp tạo nút
        if knot_method == "not_a_knot":
            self.knots = not_a_knot_vector(self.t, self.k)
        else:
            raise ValueError(f"Unsupported knot method: {knot_method}")

        self.basis = self._construct_basis()

    def _construct_basis(self):
        n_basis = len(self.knots) - self.k  # Số lượng hàm cơ sở (spline coefficients)
        basis = []
        for i in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[i] = 1.0
            basis.append(BSpline(self.knots, coeff, self.k - 1))  # SciPy dùng degree = order - 1
        return basis

    def design_matrix(self, t_eval):
        """
        Ma trận thiết kế: mỗi hàng là [B0(t), B1(t), ..., Bn(t)] tại từng t trong t_eval
        """
        t_eval = np.asarray(t_eval)
        return np.column_stack([b(t_eval) for b in self.basis])
