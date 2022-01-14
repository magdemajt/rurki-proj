import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass()
class Point:
    x: float
    y: float


def line_from_points(point_A: Point, point_B: Point):
    a = (point_B.y - point_A.y) / (point_B.x - point_A.x)

    b = point_A.y - a * point_A.x

    return a, b


def get_function_from_x(x, i, base_dim, func_range):
    real_dim = base_dim + 1
    a, b = 1, 0
    center = Point(i / real_dim * func_range, 1)
    if x > center.x:
        a, b = line_from_points(center, Point((i + 1) * func_range / real_dim, 0))
    elif x <= center.x:
        a, b = line_from_points(Point((i - 1) * func_range / real_dim, 0), center)
    return a * x + b


def get_base_function_slope(index, base_dim, func_range):
    real_dim = base_dim + 1
    return lambda x: 0 if x > (index + 1) * func_range / real_dim or x * func_range < (
                index - 1) / real_dim else get_function_from_x(x, index,
                                                               base_dim, func_range)


def main(base_dim):
    pass


if __name__ == '__main__':
    dim = 2
    x = np.arange(0, 3, 0.001)
    func = get_base_function_slope(1, dim, 3)
    plt.plot(x, np.array([func(i) for i in x]))
    plt.show()
    print('test')
