


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
    verbose = 1

    def __init__(self, f, d, x: [float], alpha: float = 0.001,
                 max_iter: int = 1000, min_decrease: int = 0.1, verbose: int = 5):
        self.f = f
        self.x = x
        self.d = d
        self.alpha = alpha
        self.max_iter = max_iter
        self.min_decrease = min_decrease
        self.verbose = verbose

    def run_descent(self):
        val = self.f(self.x)
        # Just dummy to start loop.
        val2 = val - 1
        plot = {'x': [], 'y': []}
        print("iteration {} for value of x {} with f {} and decrease {}"
              .format(self.iter, self.x, val2, (val - val2)))
        while val2 < val:
            try:
                ders = self.take_step()
                val = val2
                val2 = self.f(self.x)
                self.iter += 1
                plot['y'].append(val2)
                plot['x'].append(self.iter)
                if self.iter % self.verbose == 0:
                    print("iteration {} for value of x {} with f {} and decrease {} with derivative {}"
                          .format(self.iter, self.x, val2, (val - val2), ders))
                if ((val - val2) < self.min_decrease) |\
                        (self.iter == self.max_iter):
                    break
            except OverflowError:
                print("Limits reached, seems the equation doesn't have a minima.")
                break

        # Minima reached.
        print("Final - iteration {} for value of x {} with f {} and decrease {}"
              .format(self.iter, self.x, val2, (val - val2)))
        return [val, self.x, plot]

    def take_step(self):
        updated_x = []
        ders = []
        for i in range(len(self.x)):
            der = self.d(self.x, i)
            ders.append(der)
            updated_x.append(float(self.x[i]) - self.alpha * float(der))
        self.x = updated_x
        return ders

