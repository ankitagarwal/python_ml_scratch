


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

    def __init__(self, f, d, x: [float], alpha: float = 0.001, max_iter: int = 1000, min_decrease: int = 0.1):
        self.f = f
        self.x = x
        self.d = d
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_decrease = min_decrease

    def run_descent(self):
        val = self.f(self.x)
        # Just dummy to start loop.
        val2 = val - 1
        plot = {'x': [], 'y': []}
        while val2 < val:
            self.take_step()
            val = val2
            val2 = self.f(self.x)
            self.iter += 1
            plot['y'].append(val2)
            plot['x'].append(self.iter)
            if self.iter % 5 == 0:
                print("iteration ", self.iter)
                print("x ", self.x)
                print("function value ", val2)
                print("Decrease ", val - val2)
            if ((val - val2) < self.min_decrease) |\
                    (self.iter == self.max_iter):
                break

        # Minima reached.
        return [val, self.x, plot]

    def take_step(self):
        der = self.d(self.x)
        self.x = [float(v) - self.alpha * der for v in self.x]

