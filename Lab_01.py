import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train={x_train}")
print(f"y_train={y_train}")

m = x_train.shape[0]

m = len(x_train)

i = 0

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.show()
