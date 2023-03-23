import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def visualize_boxes(volume, mask, centroids):
    z = volume.shape[2]
    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()

    # print(f"mask dtype : {mask.dtype} , ct dtype : {volume.dtype}")
    masked = np.ma.masked_where(mask == 0, mask)
    im = axs[0].imshow(volume[:, :, z//2], cmap="gray")
    im_masked = np.ma.masked_array(volume, masked)
    msk = axs[1].imshow(im_masked[:, :, z//2], cmap="gray")
    for i, coords in enumerate(centroids[z//2], start=2):
        axs[i].imshow(volume[
            coords["bbox"]["y1"]:coords["bbox"]["y2"],
            coords["bbox"]["x1"]:coords["bbox"]["x2"],
            z//2
        ], cmap="gray")

    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.04])
    slider = Slider(slider_ax, 'Index', 1, z, valinit=z//2, valstep=1)

    # Define a function to update the displayed image when the slider is moved
    def update(val):
        # Get a reference to the centroid plot
        index = int(slider.val)
        im.set_data(volume[:, :, index])
        msk.set_data(im_masked[:, :, index])

        for i, coords in enumerate(centroids[index], start=2):

            axs[i].imshow(volume[
                coords["bbox"]["y1"]:coords["bbox"]["y2"],
                coords["bbox"]["x1"]:coords["bbox"]["x2"],
                index
            ], cmap="gray")
        fig.canvas.draw()

    # Register the update function with the slider
    slider.on_changed(update)

    # Show the plot
    plt.show()


def visualize_windowing(volume, windowed, filtered=None):
    z = volume.shape[2]
    fig, axs = plt.subplots(1, 2)
    axs = axs.flatten()
    axs[0].title.set_text("Volume")
    axs[1].title.set_text("Windowed Volume")
    # axs[2].title.set_text("Filtered Volume")
    print(
        f"mask shape, dtype : {windowed.shape},{windowed.dtype} , ct shape, dtype : {windowed.shape},{volume.dtype}")
    im = axs[0].imshow(volume[:, :, z//2], cmap="gray")
    win = axs[1].imshow(windowed[:, :, z//2], cmap="gray")
    # fil = axs[2].imshow(filtered[:, :, z//2], cmap="gray")

    slider_ax = plt.axes([0.2, 0.02, 0.6, 0.04])
    slider = Slider(slider_ax, 'Index', 1, z, valinit=z//2, valstep=1)

    # Define a function to update the displayed image when the slider is moved
    def update(val):
        # Get a reference to the centroid plot
        index = int(slider.val)
        im.set_data(volume[:, :, index])
        win.set_data(windowed[:, :, index])
        # fil.set_data(filtered[:, :, index])

        fig.canvas.draw()

    # Register the update function with the slider
    slider.on_changed(update)
    axs[0].axis('off')
    axs[1].axis('off')
    # axs[2].axis('off')

    # Show the plot
    plt.show()
