class CostFunctions:

    @staticmethod
    def square_sum(x):
        total = 0
        for vec in x:
            total += vec*vec
        return total

    @staticmethod
    def der_square_sum(x):

        total = 0
        for vec in x:
            total += vec
        total = 2 * total
        return total

    def square_error(self):
        pass