from old_codes.models.LogisticRegression import LogisticRegression as LR
import pandas as pd
import matplotlib.pyplot as plt

X = [
        [2.7810836, 2.550537003],
        [1.465489372, 2.362125076],
        [3.396561688, 4.400293529],
        [1.38807019, 1.850220317],
        [3.06407232, 3.005305973],
        [7.627531214, 2.759262235],
        [5.332441248, 2.088626775],
        [6.922596716, 1.77106367],
        [8.675418651, -0.242068655],
        [7.673756466, 3.508563011]
]
y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

model = LR()
model.train(X, y)
df = pd.DataFrame(model.predict_proba(X), columns=['pred'])
df.reset_index().plot.scatter(y='pred', x='index')  # Not the right way to plot 1-d data.
plt.show()
print(model.predict(X))
