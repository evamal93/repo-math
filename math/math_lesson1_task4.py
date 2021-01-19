import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return np.cos(1 * x)
diap = np.linspace(0, 3, 51)
y = f(diap)
plt.plot(diap, y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def f(x):
    return np.cos(4 * x)
diap = np.linspace(0, 3, 51)
y = f(diap)
plt.plot(diap, y)
plt.show()