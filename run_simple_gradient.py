import pandas as pd
import matplotlib.pyplot as plt
from models import *

f = CostFunctions.square_sum
d = CostFunctions.der_square_sum


# Simple uni variate square function.
x = [[v] for v in range(-50, 50)]
Helper.run_gradient_descent(f, d, x, [5000], 0.1, 1)

# Let's try another eq, y = a^2 + b^2
x = [[x1, x2] for x1 in range(-50, 50) for x2 in range(10, 20)]
# Let's see if our gradient works for this, obviously the answer should be at point 0.
Helper.run_gradient_descent(f, d, x, [5000], 0.1, 1)

