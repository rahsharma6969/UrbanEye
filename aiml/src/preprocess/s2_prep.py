import numpy as np
import rioxarray as rxr

GOOD_SCL = {4,5,6,7,11}  # keep vegetation/bare/water/shadow-ish; exclude clouds 8/9/10

def cloud_mask_from_scl(scl_da):
    data = scl_da.data
    keep = np.isin(data, list(GOOD_SCL))
    return keep  # True=keep

def apply_mask(arr, mask):
    # arr shape [C,H,W], mask [H,W]
    masked = arr.copy()
    for c in range(masked.shape[0]):
        band = masked[c]
        band[~mask] = np.nan
        masked[c] = band
    return masked
