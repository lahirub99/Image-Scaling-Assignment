import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
#from glob import glob  # for reading in files

image = cv2.imread('images/dunithi.jpg')

print(image.shape)      # output : (height, width, channels) = (h=903, w=902, 3)

fig, ax = plt.subplots( figsize=(10,10) )
ax.imshow(image)
#ax.axis('off')
plt.show()