import numpy as np


class CostFunctions:

    @staticmethod
    def square_sum(x):
        total = 0
        for vec in x:
            total += vec * vec
        return total

    @staticmethod
    def der_square_sum(x, i: int = 0):
        return 2 * x[i]

    @staticmethod
    def square_cube_sum(x):
        return (x[0] ** 3) + (x[1] ** 2)

    @staticmethod
    def der_sqaure_cube_sum(x, i):
        # 3a^2
        if i == 0:
            return 3*(x[0] ** 2)
        elif i == 1:
            # 2b
            return 2*x[1]
        raise Exception("Invalid data supplied")


    def square_error(self):
        pass
