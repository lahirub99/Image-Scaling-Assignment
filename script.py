import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
#from glob import glob  # for reading in files

#region Section 1: Reading in images
#region Section 1.1: Reading in images using CV2
image = cv2.imread('images/original.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert to RGB color space
#endregion

#region Section 1.2: Reading in images using Matplotlib
# image_rgb = plt.imread('images/test.jpg')
#endregion

#print(image.shape)      
# output : (height, width, channels) = (h=903, w=902, 3)
#endregion


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


''' Grayscale images:'''
#region Section 3: Converting to grayscale
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)         # output : (height, width) = (h=903, w=902)

fig, ax = plt.subplots( figsize=(10,10) )
ax.imshow(image_gray, cmap='gray')
#ax.axis('off') # visibility of x- and y-axes
ax.grid(True)   # show gridlines
plt.show()

#endregion


''' Resizing images:'''
#region Section 4: Resizing images
#region Section 4.1: Resizing method 01
image_resized = cv2.resize(image_rgb, None, fx=0.7, fy=0.7)
print(image_resized.shape)      # output : (height, width, channels) = (h=100, w=100, 3)
#endregion

#region Section 4.2: Resizing method 02
image_100x200 = cv2.resize(image_rgb, (100, 200))
#endregion

#region Section 4.3: Resizing method 03
image_upscaled = cv2.resize(image_resized, (200, 200), interpolation=cv2.INTER_CUBIC)
#endregion

# Plotting the image
fig, ax = plt.subplots( figsize=(5,5) )
#ax.imshow(image_resized)
#ax.imshow(image_100x200)
ax.imshow(image_upscaled)
ax.grid(True)   # show gridlines
plt.show()
#endregion


''' Applying Filters:'''
#region Section 5: Applying Filters

#region Section 5.1: Sharpening filter
kernal_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
image_sharpened = cv2.filter2D(image_rgb, -1, kernal_sharpening)
# Parameters of filter2D: src = image_rgb, ddepth = -1, kernal = kernal_sharpening
#endregion

#region Section 5.2: Blurring filter
kernal_3x3 = np.ones((3,3), np.float32) / 9
# kernal_3x3 =    [[0.11111111 0.11111111 0.11111111]
#                  [0.11111111 0.11111111 0.11111111]
#                  [0.11111111 0.11111111 0.11111111]]

image_blurred = cv2.filter2D(image_rgb, -1, kernal_3x3)
#endregion

# Plotting the image
fig, axs = plt.subplots( 1, 3, figsize=(10,5) )
axs[0].imshow(image_rgb)
axs[1].imshow(image_sharpened)
axs[2].imshow(image_blurred)

# show gridlines
# axs[0].grid(True)   
# axs[1].grid(True)
# axs[0].grid(axs[1].grid(axs[2].grid(True)))
[ax.grid(True) for ax in axs]

plt.show()
#endregion


#region Section 6: Saving images

#region Section 6.1: Saving images in BGR color space
## CV2 uses BGR color space by default, so outputs will look inverted with abnormal colors
# cv2.imwrite('images/output/output_gray.jpg', image_gray)
# cv2.imwrite('images/output/output_resized.jpg', image_resized)
# cv2.imwrite('images/output/output_100x200.jpg', image_100x200)
# cv2.imwrite('images/output/output_upscaled.jpg', image_upscaled)
# cv2.imwrite('images/output/output_sharpened.jpg', image_sharpened)
# cv2.imwrite('images/output/output_blurred.jpg', image_blurred)
#endregion

images = [image_gray, image_resized, image_100x200, image_upscaled, image_sharpened, image_blurred]

# Define the output directory
output_directory = 'images/output/output_'
output_filenames = ['gray', 'resized', "100x200", "upscaled", "sharpened", "blurred"]


#region Section 6.2: Saving images in RGB color space using CV2
# Loop through the images and save them with color space conversion
for i in range (6):
    # Construct the output file name by appending it to the output directory
    output_filename = f"{output_directory}{output_filenames[i]}{'.jpg'}"
    #     The error message you're encountering, "could not find a writer for the specified extension," typically occurs 
    #     when OpenCV can't determine the correct image format based on the file extension you provided in the output_filename.
    #     To resolve this issue, make sure you provide a file extension that is recognized by OpenCV's imwrite function. 
    #     Common image formats that are supported include .jpg, .png, .bmp, and others.   
    
    # Convert the image to RGB color space and save it
    temp = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_filename, temp)
    print(f"Image saved at {output_filename}")
#endregion


# #region Section 6.3: Saving images in RGB color space using Matplotlib
# # Loop through the images and save them with color space conversion
# for i in range (6):
#     # Construct the output file name by appending it to the output directory
#     output_filename = f"{output_directory}{output_filenames[i]}{'_mthplt.jpg'}"
    
#     ''' !!! if the image was imported in CV2 it should be convert the image to RGB color space and save it. '''

#     # Just save the image directly, no need to convert back to RGB from BGR color space as we did with CV2
#     cv2.imwrite(output_filename, images[i])
#     print(f"Image saved at {output_filename}")
# #endregion

#endregion