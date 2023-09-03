import matplotlib.pyplot as plt

''' Image plotting : (Used for testing purposes only) '''
def plot_image(image, title='', gray=False):     # gray=True for grayscale images
    fig, ax = plt.subplots( figsize=(5,5) )
    if gray:
        ax.imshow(image, cmap='gray')
    else: ax.imshow(image)
    ax.set_title(title)
    ax.axis('on')      # visibility of x- and y-axes
    ax.grid(True)     # show gridlines
    plt.tight_layout()
    plt.show()
    return
