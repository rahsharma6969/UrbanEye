import numpy as np
from scipy.ndimage import median_filter

def denoise_sar(arr, size=3):
    # simple median filter as placeholder
    return median_filter(arr, size=(1,size,size))
