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
g = GradientDescent(f, d, [5000], 0.2)
[val, x, plot] = g.run_descent()
print(plot)
pd.DataFrame(plot).plot.scatter('x', 'y')
plt.show()

# Let's try another eq, y = a^2 + b^2

df = pd.DataFrame([f([x1, x2]) for x1 in range(-50, 50) for x2 in range(10, 20)])
df.plot.hist()
plt.show()

df = pd.DataFrame([x1, x2] for x1 in range(-50, 50) for x2 in range(10, 20))
df.plot.scatter(df.columns[0], df.columns[1])
plt.show()

# Derivative of this is 2(x1 +x2) , a linear function
df = pd.DataFrame([d([x1, x2]) for x1 in range(-50, 50) for x2 in range(10, 20)])
df.plot.hist()
plt.show()

# Let's see if our gradient works for this, obviously the answer should be at point 0.
g = GradientDescent(f, d, [5000, -3000], 0.2, verbose=1)
[val, x, plot] = g.run_descent()
pd.DataFrame(plot).plot.scatter('x', 'y')
plt.show()



