import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Union


@dataclass()
class Point:
    x: float
    y: float

@dataclass()
class Function:
    a: float
    b: float

    def get_value(self, x):
        return self.a * x + self.b

    def get_slope(self, x):
        return self.a

    def get_start(self):
        return 0

    def get_end(self):
        return 3


A = 4 * 3.14 * 6.67
u_schlange = Function(-1 / 3, 5)


def get_function_from_x(x, i, base_dim, func_range):
    real_dim = base_dim
    a, b = 1, 0
    center = Point(i / real_dim * func_range, 1)
    if x > center.x:
        a, b = line_from_points(center, Point((i + 1) * func_range / real_dim, 0))
    elif x <= center.x:
        a, b = line_from_points(Point((i - 1) * func_range / real_dim, 0), center)
    return Function(a, b)


class BaseFunction:
    index: int
    base_dim: int
    func_range: float

    def __init__(self, index, base_dim, func_range):
        self.index = index
        self.base_dim = base_dim + 1
        self.func_range = func_range

    def get_function_for_x(self, x):
        return get_function_from_x(x, self.index, self.base_dim, self.func_range)

    def is_zero(self, x):
        return x > (self.index + 1) * self.func_range / self.base_dim or x < self.func_range * (
                self.index - 1) / self.base_dim

    def get_start(self):
        return self.func_range * (
                self.index - 1) / self.base_dim

    def get_end(self):
        return (self.index + 1) * self.func_range / self.base_dim

    def get_value(self, x):
        if self.is_zero(x):
            return 0
        return self.get_function_for_x(x).get_value(x)

    def get_slope(self, x):
        if self.is_zero(x):
            return 0
        return self.get_function_for_x(x).a


def line_from_points(point_A: Point, point_B: Point):
    a = (point_B.y - point_A.y) / (point_B.x - point_A.x)

    b = point_A.y - a * point_A.x

    return a, b


def b_u_j(base_function_slope_lambda_u: Union[BaseFunction, Function],
          base_function_slope_lambda_v: Union[BaseFunction, Function]):
    func = lambda x: -base_function_slope_lambda_u.get_slope(x) * base_function_slope_lambda_v.get_slope(x)
    return quad(func, base_function_slope_lambda_u.get_start(), base_function_slope_lambda_v.get_end())[0]


def l(v: Union[BaseFunction, Function]):
    func = lambda x: v.get_slope(x)
    return A * quad(func, 1, 2)[0]


def rhs_matrix_coeff(e):
    return l(e) - b_u_j(u_schlange, e)


def main(base_dim, func_range):
    B = np.array(
        [[b_u_j(BaseFunction(i, base_dim, func_range), BaseFunction(j, base_dim, func_range)) for i in
          range(base_dim)] for j in range(base_dim)],
        dtype="float64"
    )
    print(B)

    R = np.array(
        [rhs_matrix_coeff(BaseFunction(index, base_dim, func_range)) for index in range(base_dim)],
        dtype="float64"
    )

    print(R)

    res = np.linalg.solve(B, R)

    print(res)

    res_func = lambda x: sum(res[i] * BaseFunction(i, base_dim, func_range).get_value(x) for i in range(len(res))) + u_schlange.get_value(x)
    x = np.arange(0, 3, 0.001)
    plt.plot(x, np.array([res_func(i) for i in x]))
    plt.show()


if __name__ == '__main__':
    main(125, 3)
