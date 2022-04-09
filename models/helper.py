import pandas as pd
import matplotlib.pyplot as plt

from models import GradientDescent


class Helper:
    @staticmethod
    def run_gradient_descent(f, d, x, start, alpha, verbose, graphs: bool = False):
        df = pd.DataFrame([f(v) for v in x])
        if graphs:
            df.plot.hist()
            plt.show()

        # Derivative
        df = pd.DataFrame([d(v) for v in x])
        if graphs:
            df.plot.hist()
            plt.show()

        # Let's see if our gradient works for this, obviously the answer should be at point 0.
        g = GradientDescent(f, d, start, alpha, verbose=verbose)
        [val, x, plot] = g.run_descent()
        if graphs:
            pd.DataFrame(plot).plot.scatter('x', 'y')
            plt.show()
