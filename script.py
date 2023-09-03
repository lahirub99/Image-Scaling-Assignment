import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
#from glob import glob  # for reading in files

image = cv2.imread('images/dunithi.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB color space

#print(image.shape)      
# output : (height, width, channels) = (h=903, w=902, 3)


#region Section X: Plotting the image

fig, axs = plt.subplots(2, 4, figsize=(15,5) )
axs[0][0].imshow(image_rgb)
axs[0][1].imshow(image_rgb[:,:,0], cmap='Reds')
axs[0][2].imshow(image_rgb[:,:,1], cmap='Greens')
axs[0][3].imshow(image_rgb[:,:,2], cmap='Blues')
#axs.axis('off') # visibility of x- and y-axes
#axs.grid(True)   # show gridlines

plt.show()

#endregion


'''
#region Section X: Plotting the image

fig, ax = plt.subplots( figsize=(10,10) )
ax.imshow(image_rgb)
#ax.axis('off') # visibility of x- and y-axes
ax.grid(True)   # show gridlines

plt.show()

#endregion
'''