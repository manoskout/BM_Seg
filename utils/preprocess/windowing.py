import numpy as np
import pydicom as dcm
import scipy
import pywt
import skimage.restoration as restoration
from typing import Tuple


def crop_pad_vol(vol:np.ndarray, mask: np.ndarray, crop_size:tuple = (384,384))-> Tuple[np.ndarray, np.ndarray]:

    cropx, cropy = crop_size
    h, w, z = vol.shape
    starty = startx = 0

    # Crop only if the crop size is smaller than image size
    if cropy <= h:
        starty = h//2-(cropy//2)

    if cropx <= w:
        startx = w//2-(cropx//2)

    cropped_img = vol[starty:starty+cropy, startx:startx+cropx, :]
    cropped_msk = mask[starty:starty+cropy, startx:startx+cropx, :]


    # Add padding, if the image is smaller than the desired dimensions
    old_image_height, old_image_width, old_image_z = cropped_img.shape
    new_image_height, new_image_width = old_image_height, old_image_width

    if old_image_height < cropy:
        new_image_height = cropy
    if old_image_width < cropy:
        new_image_width = cropy

    if (old_image_height != new_image_height) or (old_image_width != new_image_width):

        padded_img = np.full(
            (new_image_height, new_image_width), 0, dtype=np.float32)
        padded_msk = np.full(
            (new_image_height, new_image_width), 0, dtype=np.float32)

        x_center = (new_image_height - old_image_width) // 2
        y_center = (new_image_width - old_image_height) // 2

        padded_img[y_center:y_center+old_image_height,
                    x_center:x_center+old_image_width, :] = cropped_img
        padded_msk[y_center:y_center+old_image_height,
                    x_center:x_center+old_image_width, :] = cropped_img

        # print('Padded: ',padded_img.shape)
        new_vol, new_mask = padded_img, padded_msk

    else:
        new_vol, new_mask = cropped_img, cropped_msk
    # print("IMAGE_SHAPE: ",new_sample['image'].shape)
    return new_vol, new_mask

def check_volume(vol):
    if type(vol) == np.ndarray:
        return vol
    elif type(vol) == str:
        return dcm.dcmread(vol).pixel_array
    elif type(vol) == list:
        return np.array([dcm.dcmread(sl).pixel_array for sl in vol])


class CTWindowing:
    """
    A CT (computed tomography) windowing filter used for image preprocessing in medical imaging.

    CT windowing is the process of assigning Hounsfield units (HU) to a specific range of values in the image, 
    which are then mapped to a displayable gray-scale range. 

    Attributes:
        window (str): The window to use for windowing. Possible options include: 'abdomen', 'angio', 'bone',
                      'temporal_bones', 'soft_tissue', 'brain', 'mediastinum', 'lungs', 'test'.
        intercept (int): The intercept value for the CT windowing formula. 
        slope (int): The slope value for the CT windowing formula.
        window_center (int): The center value for the window range.
        window_width (int): The width of the window range.

    Methods:
        volume_windowing(image, rescale=True):
            Apply CT windowing to the input image.

    """

    def __init__(self, window, intercept, slope):
        """
        Initialize a new instance of the CTWindowing class.

        Args:
            window (str): The window to use for windowing. Possible options include: 'abdomen', 'angio', 'bone',
                          'temporal_bones', 'soft_tissue', 'brain', 'mediastinum', 'lungs', 'test'.
            intercept (int): The intercept value for the CT windowing formula. 
            slope (int): The slope value for the CT windowing formula.

        Raises:
            ValueError: If an invalid window value is specified.
        """
        win_dict = {'abdomen':
                    {'wl': 60, 'ww': 400},
                    'angio':
                    {'wl': 300, 'ww': 600},
                    'bone':
                    {'wl': 400, 'ww': 1800},
                    'temporal_bones':
                    {'wl': 600, 'ww': 2800},
                    'soft_tissue':
                    {'wl': 50, 'ww': 250},
                    'brain':
                    {'wl': 40, 'ww': 80},
                    'mediastinum':
                    {'wl': 50, 'ww': 350},
                    'lungs':
                    {'wl': -600, 'ww': 1500},
                    'test':
                    {'wl': 40, 'ww': 400}
                    }
        self.window = window
        self.intercept = intercept
        self.slope = slope
        if self.window in win_dict.keys():
            self.window_center, self.window_width = win_dict[
                self.window]["wl"], win_dict[self.window]["ww"]
        else:
            return ValueError("Unspecified window value")

    def volume_windowing(self, image, rescale=True):
        """
        Apply CT windowing to the input image.

        Args:
            image (numpy.ndarray): The input image to apply CT windowing to.
            rescale (bool, optional): Whether to rescale the image values to the range [0, 1]. Defaults to True.

        Returns:
            numpy.ndarray: The windowed image.
        """
        # apply CT windowing to the image
        image = (image*self.slope + self.intercept)
        image = np.clip(image, self.window_center - (self.window_width/2),
                        self.window_center + (self.window_width/2))
        image = (image - self.window_center) / (self.window_width/2)

        # convert the image and mask to PyTorch tensors
        if rescale:
            # print("image before scalling ", image.max())
            min_hu = image.min()
            max_hu = image.max()
            img_rescaled = (image - min_hu) / (max_hu - min_hu) * 255
            # print("image after scalling ", image.max())

            # normalize to 0,1
            img = (img_rescaled - img_rescaled.min()) / \
                (img_rescaled.max() - img_rescaled.min())

            return img

        return img


class Preprocessing:
    """
    A class for preprocessing medical imaging data.

    Attributes:
    ----------
    window: tuple of int
        The Hounsfield unit (HU) window to use for CT windowing.
    metadata: dict
        The metadata associated with the medical imaging data.
    rescale: bool, optional
        Whether or not to rescale the pixel values to HU (default=True).

    Methods:
    -------
    gaussian_filter(volume, sigma=1, size=3)
        Apply a Gaussian filter to the input volume.
    """

    def __init__(self, window, metadata, rescale=True) -> None:
        """
        Initializes the Preprocessing object with the specified HU window, metadata,
        and rescaling option.

        Parameters:
        ----------
        window: tuple of int
            The HU window to use for CT windowing.
        metadata: dict
            The metadata associated with the medical imaging data.
        rescale: bool, optional
            Whether or not to rescale the pixel values to HU (default=True).
        """

        self.intercept = int(metadata["RescaleIntercept"])
        self.slope = int(metadata["RescaleSloce"])
        self.rescale = rescale

        # TODO -> change it into a function
        # Initialize a CTWindowing object with the specified window, intercept, and slope.
        self.windowing = CTWindowing(
            window, intercept=self.intercept, slope=self.slope)

    def gaussian_filter(self, volume, sigma=1, size=3) -> np.ndarray:
        """
        Applies a Gaussian filter to the input volume.

        Parameters:
        ----------
        volume: numpy array
            The volume to apply the filter to.
        sigma: float, optional
            The standard deviation of the Gaussian kernel (default=1).
        size: int, optional
            The size of the Gaussian kernel (default=3).

        Returns:
        -------
        smoothed_volume: numpy array
            The volume with the Gaussian filter applied.
        """

        # Construct the Gaussian kernel
        kernel = np.zeros((size, size, size))
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    x = i - size // 2
                    y = j - size // 2
                    z = k - size // 2
                    kernel[i, j, k] = np.exp(-(x ** 2 +
                                             y ** 2 + z ** 2) / (2 * sigma ** 2))
        # Normalize the kernel
        kernel /= np.sum(kernel)

        # Apply the Gaussian filter using the convolve function from the SciPy package
        smoothed_volume = scipy.ndimage.convolve(volume, kernel)
        return smoothed_volume

    def median_filter(self, volume, size=3) -> np.ndarray:
        return scipy.ndimage.median_filter(volume, size)

    def wavelet_filter(self, volume, wavelet='db4', level=3):

        # Perform the wavelet decomposition
        coeffs = pywt.wavedecn(volume, wavelet, level=level)
        # Reconstruct the denoised volume from the thresholded coefficients
        denoised_volume = pywt.waverecn(coeffs, wavelet)
        return denoised_volume

    