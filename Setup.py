import numba
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

print('hi')
print('test2')

x = np.array([1, 2, 3, 4, 10, 20, 30, 40])
y = np.sum(x)
print(y)

plt.plot(x)
plt.title('Sample Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()