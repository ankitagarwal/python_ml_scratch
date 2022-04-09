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

    def square_error(self):
        pass
