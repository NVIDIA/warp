import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np


fig = plt.figure()


def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

a = np.random.rand(128, 128)

im = plt.imshow(a, animated=True)


def updatefig(*args):
    global x, y
    x += np.pi / 15.
    y += np.pi / 20.
    a = np.random.rand(128, 128)
    im.set_array(a)
    return im,

ani = animation.FuncAnimation(fig, updatefig, interval=16, blit=True)
plt.show()