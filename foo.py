import numpy as np
from skimage.draw import circle
from tensorflow_addons.image import connected_components

x = np.zeros((100, 100))
rr, cc = circle(25, 25, 20)
x[rr, cc] = 1
rr, cc = circle(75, 75, 20)
x[rr, cc] = 1

y = connected_components(x)
