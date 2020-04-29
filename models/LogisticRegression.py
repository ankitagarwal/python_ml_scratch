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
    trained = False

    def linear_equation(self, x: [int]) -> int:
        val = self.coefs[0]
        if len(self.coefs) != (len(x) + 1):
            raise RuntimeError ("Mismatch between length of coefficients and data. "
                                "Expected {}, found {}".format(len(self.coefs), len(x)+1))
        for i in range(len(self.coefs)-1):
            val += self.coefs[i + 1] * x[i]
        return val

    def sigmoid(self, x: int) -> int:
        return 1/(1 + np.exp(-x))

    def predict_single(self, x: [int]) -> int:
        """
        Predict a single value. Basically this is sigmoid over linear.
        """
        return self.sigmoid(self.linear_equation(x))

    def predict(self, X: [[int]], boundary: int = 0.5) -> [[int]]:
        probabilities = self.predict_proba(X)
        return [1 if i > boundary else 0 for i in probabilities]

    def predict_proba(self, X: [[int]]) -> [[int]]:
        if not self.trained:
            raise RuntimeError("Can't predict without being trained.")
        Y = []
        X = np.array(X)
        l, m = X.shape
        for i in range(X.shape[0]):
            x = X[i]
            Y.append(self.predict_single(x))
        return Y

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

    def gradient(self, y_hat: int, y: int, x: [int]) -> [int]:
        """
        see:
            images/cross-entropy-gradient.png
        """
        x = list(x)
        x.insert(0, 1)  # Constant term.
        return [i * (y_hat - y) for i in x]

    def train(self, X_train: [[int]], Y_train: [[int]], epoch: int = 1000, lr: int = 0.001):
        # Estimate logistic regression coefficients using stochastic gradient descent
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        self.coefs = np.zeros(X_train.shape[1]+1)
        for epoch in range(epoch):
            sum_error = 0
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]
                y_hat = self.predict_single(x)
                sum_error += self.cost_function(y_hat, y)
                gradient = self.gradient(y_hat, y, x)
                self.coefs = [lr*i for i in gradient]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, lr, sum_error))
        self.trained = True
        return self.coefs


