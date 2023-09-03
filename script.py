import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
#from glob import glob  # for reading in files

image = cv2.imread('images/dunithi.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB color space

#print(image.shape)      
# output : (height, width, channels) = (h=903, w=902, 3)


''' Displaying the image:
#region Section X: Plotting the image

fig, ax = plt.subplots( figsize=(10,10) )
ax.imshow(image_rgb)
#ax.axis('off') # visibility of x- and y-axes
ax.grid(True)   # show gridlines

plt.show()

#endregion
'''


''' Splitting the image into RGB channels: 
#region Section 2: Plotting the image

fig, axs = plt.subplots(2, 4, figsize=(15,15) )
axs[0][0].imshow(image_rgb)
axs[0][1].imshow(image_rgb[:,:,0], cmap='Reds')
axs[0][2].imshow(image_rgb[:,:,1], cmap='Greens')
axs[0][3].imshow(image_rgb[:,:,2], cmap='Blues')
axs[1][0].imshow(image)
axs[1][1].imshow(image[:,:,0], cmap='Blues')
axs[1][2].imshow(image[:,:,1], cmap='Greens')
axs[1][3].imshow(image[:,:,2], cmap='Reds')
#axs.axis('off') # visibility of x- and y-axes
#axs.grid(True)   # show gridlines
for i in range(2):
    for j in range(4):
        axs[i][j].axis('on')
        axs[i][j].grid(True)

plt.show()

#endregion
'''


''' Grayscale images:
#region Section 3: Converting to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)         # output : (height, width) = (h=903, w=902)

fig, ax = plt.subplots( figsize=(10,10) )
ax.imshow(image_gray, cmap='gray')
#ax.axis('off') # visibility of x- and y-axes
ax.grid(True)   # show gridlines
plt.show()

#endregion
'''

''' Resizing images:
#region Section 4: Resizing images
#region Section 4.1: Resizing method 01
image_resized = cv2.resize(image_rgb, None, fx=0.25, fy=0.25)
print(image_resized.shape)      # output : (height, width, channels) = (h=100, w=100, 3)
#endregion

#region Section 4.2: Resizing method 02
image_100x200 = cv2.resize(image_rgb, (100, 200))
#endregion

#region Section 4.3: Resizing method 03
image_upcaled = cv2.resize(image_rgb, (2000, 2000), interpolation=cv2.INTER_CUBIC)
#endregion

# Plotting the image
fig, ax = plt.subplots( figsize=(5,5) )
#ax.imshow(image_resized)
#ax.imshow(image_100x200)
ax.imshow(image_upcaled)
ax.grid(True)   # show gridlines
plt.show()
#endregion
'''

''' Applying Filters:'''
#region Section 5: Applying Filters

#region Section 5.1: Sharpening filter
kernal_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
image_sharpened = cv2.filter2D(image_rgb, -1, kernal_sharpening)
# Parameters of filter2D: src = image_rgb, ddepth = -1, kernal = kernal_sharpening
#endregion



# Plotting the image
fig, axs = plt.subplots( 1, 2, figsize=(10,5) )
axs[0].imshow(image_rgb)
axs[0].grid(True)   # show gridlines
axs[1].imshow(image_sharpened)
axs[1].grid(True)   # show gridlines
plt.show()
#endregion
