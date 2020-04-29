import numpy as np

class LogisticRegression:

    """
    Module to implement Logistic Regression using stochastic Gradient Descent. Not using numpy products was an intent.
    See: images/sigmoid
    """

    coefs = []

    def linear_equation(self, coefs: list[int], x: [int]) -> int:
        val = coefs[0]
        if len(coefs) != (len(x) + 1):
            raise RuntimeError ("Mismatch between length of coefficients and data. "
                                "Expected {}, found {}".format(len(coefs), len(x)))
        for i in range(len(coefs)-1):
            val += coefs[i + 1] * x
        return val

    def predict_single(self, x: [int]) -> int:
        """
        Predict a single value. Basically this is sigmoid over linear.
        """
        linear = self.predict_single(x)
        y_hat = 1/1 + e^(-linear)
        return y_hat
