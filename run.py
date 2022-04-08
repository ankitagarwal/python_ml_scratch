import pandas as pd
import matplotlib.pyplot as plt
from models import *

f = CostFunctions.square_sum
d = CostFunctions.der_square_sum


# Simple uni variate square function.
df = pd.DataFrame([f([x]) for x in range(-50, 50)])
df.plot.hist()
plt.show()

# Derivative of this is 2x , a linear function
df = pd.DataFrame([d([x]) for x in range(-50, 50)])
df.plot.hist()
plt.show()

# Let's see if our gradient works for this, obviously the answer should be at point 0.
g = GradientDescent(f, d, [0])


