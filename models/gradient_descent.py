


# We have the objective/loss function J(x1, x2,...)
# Which we want to minimize for now.
# xn := xn - a*derivative(J(x1,x2,..xn))
# Square error cost function


class GradientDescent:

    f = None
    x = []
    alpha = 0
    d = None
    iter = 0
    max_iter = 1000

    def __init__(self, f, d, x: [float], alpha: float = 0.001, max_iter: int = 1000):
        self.f = f
        self.x = x
        self.d = d
        self.alpha = alpha
        self.max_iter = max_iter

    def run_descent(self):
        val = self.f(self.x)
        val2 = val
        while val2 <= val:
            self.take_step()
            val2 = self.f(self.x)
            self.iter += 1
            if self.iter % 10 == 0:
                print("iteration ", self.iter)
            if self.iter == self.max_iter:
                break

        # Minima reached.
        return [val, self.x]

    def take_step(self):
        der = self.d(self.x)
        self.x -= self.alpha * der

