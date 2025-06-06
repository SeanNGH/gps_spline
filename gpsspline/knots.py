import numpy as np

def uniform_knot_vector(t, k):
    n = len(t)
    return np.concatenate((
        [t[0]] * (k - 1),
        np.linspace(t[0], t[-1], n - (k - 1) + 1),
        [t[-1]] * (k - 1)
    ))

def not_a_knot_vector(t, k):
    n = len(t)
    if k % 2 == 0:  # bậc chẵn
        s = k // 2
        knots = []
        for m in range(n + k + 1):
            if m <= s:
                knots.append(t[0])
            elif m <= n + s - 1:
                knots.append(t[m - s])
            else:
                knots.append(t[-1])
        return np.array(knots)
    else:  # bậc lẻ
        s = (k + 1) // 2
        knots = []
        for m in range(n + k + 1):
            if m <= s - 1:
                knots.append(t[0])
            elif m <= n + s - 1:
                knots.append(t[m - s + 1])
            else:
                knots.append(t[-1])
        return np.array(knots)

def even_knot_vector(t, k):
    return np.linspace(t[0], t[-1], len(t) + k)

def odd_knot_vector(t, k):
    """
    Đặt nút đều cho k lẻ (không phải not-a-knot).
    """
    return np.concatenate(([t[0]] * k, np.linspace(t[0], t[-1], len(t) - k + 1), [t[-1]] * k))
