
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np

rng = np.random.default_rng(1)
x = 10 * rng.random(50)
y = 2 * x - 5 + rng.normal(0,1,50)
plt.scatter(x, y);

plt.show()