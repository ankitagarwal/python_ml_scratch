import numpy as np


class LogisticRegression:

    """
    Module to implement Logistic Regression using stochastic Gradient Descent. Not using numpy products was an intent.
    Note: https://becominghuman.ai/machine-learning-series-day-2-logistic-regression-144af00f6ff5
    See:
        images/sigmoid.png
        images/logistic_regression_sigmoid_w_threshold.png
        images/sigmoid-proof.png
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
        y_hat = 1/1 + np.exp(-linear)
        return y_hat

    def cost_function(self, y_hat: int, y: int) -> int:
        """
        We use cross entropy log-loss because of the reasons below instead of MSE.
        Note:
            If our cost function has many local minimums (MSE in this case),
            gradient descent may not find the optimal global minimum.
        Note:
            The key thing to note is the cost function penalizes confident and wrong predictions more than
            it rewards confident and right predictions! The corollary is increasing prediction accuracy
            (closer to 0 or 1) has diminishing returns on reducing cost due to the logistic nature of
            our cost function.
        see:
            images/lr_cost_fucn_0.png
            images/lr_cost_fucn_1.png
            images/lr_cost_fucn_2.png
        """
        first_term = y * np.log(y_hat)
        second_term = (1 - y) * np.log(1 - y_hat)
        cost = -(first_term + second_term)
        return cost

    def average_cost(self, y_hats: [int], ys: [int]) -> int:
        if len(y_hats) != len(ys):
            raise RuntimeError ("Mismatch between length of predictions and data. "
                                "Expected {}, found {}".format(len(y_hats), len(ys)))
        cost = 0
        for i in range(len(y_hats)):
            cost += self.cost_function(y_hats[i], ys[i])
        cost /= len(y_hats)
        return cost