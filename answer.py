import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cv2
#from glob import glob  # for reading in files


''' Plot the image '''
def plot_image(image, title=''):
    fig, ax = plt.subplots( figsize=(10,10) )
    ax.imshow(image)
    ax.set_title(title)
    # ax.axis('on')  # visibility of x- and y-axes
    ax.grid(True)   # show gridlines
    plt.show()
    return

''' Saving a image in the disk '''
def save_image(path, image):
    # Convert the image to RGB color space since CV2 uses BGR color space by default, so outputs will look inverted with abnormal colors. 
    temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Save the image
    cv2.imwrite(path, temp)
    print(f"Image saved at {path}")
    return


# 1: Reading in images
image_path = 'images/original.jpg'
image = cv2.imread(image_path)
print('Image successfully read in: ', image_path)

# CV2 uses BGR color space by default. 
# Therefore, we need to convert to RGB color space.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

print('image_rgb.shape: ', image_rgb.shape)      #output : (height, width, channels) = (h=200, w=200, 3)




plot_image(image_rgb, 'Original Image')

# Save the image
cv2.imwrite('images/output/original.jpg', image_rgb)



# # Displaying the image:
# #region Section X: Plotting the image

# fig, ax = plt.subplots( figsize=(10,10) )
# ax.imshow(image_rgb)
# #ax.axis('off') # visibility of x- and y-axes
# ax.grid(True)   # show gridlines

# plt.show()

# #endregion
