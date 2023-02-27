import sys
import timeit
import yaml
import os
import h5py as h5
from mpi4py import MPI
import numpy as np
import multiprocessing
import tomopy as tp
from skimage.transform import resize
from tomophantom.supp.artifacts import stripes as add_stripes
from larix.methods.misc import STRIPES_DETECT, STRIPES_MERGE
from utils.data_io import saveTiff, save3DTiff, loadHDF, rescale
from utils.stripe_detection import detect_stripe_larix


class TomoH5:
    """Class to wrap around a .nxs file containing tomography scan data."""

    def __init__(self, nexus_file):
        self.file = h5.File(nexus_file, 'r')
        self.data = self.file['entry1/tomo_entry/data/data']
        self.angles = self.file['entry1/tomo_entry/data/rotation_angle']
        image_key_path = 'entry1/tomo_entry/instrument/detector/image_key'
        self.image_key = self.file[image_key_path]
        self.data_indices = np.where(self.image_key[:] == 0)
        self.shape = self.data.shape
        self.flats = self.get_flats()
        self.darks = self.get_darks()

    def contains_flats(self):
        return 1 in self.image_key

    def contains_darks(self):
        return 2 in self.image_key

    def get_angles(self):
        return self.angles[self.data_indices]

    def get_flats(self):
        if not self.contains_flats():
            return None
        return self.data[self.image_key[:] == 1]

    def get_darks(self):
        if not self.contains_darks():
            return None
        return self.data[self.image_key[:] == 2]

    def get_normalized(self, item, flats=None, darks=None):
        """Get a sinogram and normalize it with flats and darks."""
        if flats is None:
            if self.flats is None:
                raise ValueError(
                    "Self contains no flats, and none were passed in. "
                    "Please pass an ndarray containing flat fields.")
            else:
                flats = self.flats
        if darks is None:
            if self.darks is None:
                raise ValueError(
                    "Self contains no darks, and none were passed in. "
                    "Please pass an ndarray containing dark fields.")
            else:
                darks = self.darks
        # Make sure flats and darks are cropped correctly
        flats = flats[item]
        darks = darks[item]
        # Normalise with flats & darks
        norm = tp.normalize(self[item], flats, darks)
        # Minus Log
        norm[norm <= 0] = 1e-9
        norm = tp.minus_log(norm)
        return norm


    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        """Item should be adjusted so that flats & darks are ignored."""
        if not (self.contains_flats() or self.contains_darks()):
            return self.data[item]
        # assumes data_indices is in ascending order
        lo = self.data_indices[0][0]
        hi = self.data_indices[0][-1]
        if type(item) == int:
            new_item = item + lo
        elif type(item) == slice:
            new_item = slice(item.start + lo if item.start is not None else lo,
                             item.stop + lo if item.stop is not None else None,
                             item.step)
        elif type(item) == tuple:
            if type(item[0]) == int:
                new_item = [item[0] + lo]
            elif type(item[0]) == slice:
                start = item[0].start + lo if item[0].start is not None else lo
                stop = item[0].stop + lo if item[0].stop is not None else None
                new_item = [slice(start,
                                  stop,
                                  item[0].step)]
            else:
                raise ValueError("Unrecognized index.")
            for i in range(1, len(item)):
                new_item.append(item[i])
            new_item = tuple(new_item)
        else:
            raise ValueError("Unrecognized index.")
        return self.data[new_item]


def new_detect_stripe_larix(data):
    """Create a binary mask showing the locations of stripes in a sinogram.
    Uses the new method developed by Daniil Kazantsev and implemented in Larix.
    Parameters:
        data : np.ndarray
            3D array containing flat-field normalised tomography data.
            Must have shape (angles, detector Y, detector X).
    Returns:
        mask : np.ndarray
            Array of same shape as `data`. Binary mask with 1 where stripes are
            located, and 0 everywhere else.
    """
    # Apply stripe detection
    print("Rescaling data to [0, 1]...")
    data = rescale(data)
    start = timeit.default_timer()
    print("Calculating stripe weights...")
    weights = STRIPES_DETECT(data, size=250, radius_size=3)
    middle = timeit.default_timer()
    print("Merging stripe weights...")
    mask = STRIPES_MERGE(weights,
                         threshold=0.63,
                         min_stripe_length=600,
                         min_stripe_depth=30,
                         min_stripe_width=22,
                         sensitivity_perc=85.0)
    stop = timeit.default_timer()
    print(f"Mask: {mask.shape}, {mask.dtype}, [{mask.min()}, {mask.max()}]")
    print(f"Times: "
          f"\tSTRIPES_DETECT:  {middle - start:5}s\n"
          f"\tSTRIPES_MERGE :  {stop - middle:5}s\n"
          f"\tTotal         :  {stop - start:5}s")
    filename = f'./mask_{sino_index:04}'
    saveTiff(mask[:, sino_index, :], filename, normalise=True)
    return mask


def createPariedWindows(data, mask, num_windows):
    """Create input/target pairs given a tomogram and its stripe location mask.
    Splits sinograms into windows, so that image sizes are smaller and the
    volume of training data is larger.
    Parameters:
        data : np.ndarray
            3D array containing flat-field normalised tomography data.
            Must have shape (angles, detector Y, detector X).
        mask : np.ndarray
            Binary mask of same shape as `data`, with 1 where stripes are and 0
            everywhere else.
        num_windows : int
            Number of windows to split a sinogram into. Total sinogram width
            must be divisible by this parameter.
    """
    # Swap axes so sinograms are in axis 0
    # i.e. data has shape (detector Y, angles, detector X)
    data = np.swapaxes(data, 0, 1)
    mask = np.swapaxes(mask, 0, 1)
    # Loop through each sinogram
    for s in range(data.shape[0]):
        # Split sinogram into 5 windows of width 512 pixels
        sino_windows = np.split(data[s], num_windows, axis=-1)
        mask_windows = np.split(mask[s], num_windows, axis=-1)
        # Loop through each window
        for w in range(num_windows):
            # if sinogram doesn't contain stripes, create input/target pair
            if mask_windows[w].sum() == 0:
                # Save 'clean' target
                clean = sino_windows[w]
                filename = f'data/clean/{s:04}_w{w:02}'
                saveTiff(clean, filename, normalise=True)
                # Add synthetic stripes to sinogram
                # currently done with TomoPhantom, other methods could be used
                stripe_type = np.random.choice(['partial', 'full'])
                stripe = add_stripes(clean, percentage=1, maxthickness=2,
                                     intensity_thresh=0.2,
                                     stripe_type=stripe_type,
                                     variability=0.005)
                # Clip back to original range
                stripe = np.clip(stripe, clean.min(), clean.max())
                # Save 'stripe' input
                filename = f'data/stripe/{s:04}_w{w:02}'
                saveTiff(stripe, filename, normalise=True)


if __name__ == '__main__':
    tomo = TomoH5('/dls/i12/data/2022/nt33730-1/rawdata/119617.nxs')
    print(f"Data Shape: {tomo.shape}")
    if tomo.contains_flats():
        print(f"Flats: {tomo.flats.shape}")
    if tomo.contains_darks():
        print(f"Darks: {tomo.darks.shape}")

    # Load normalized 3D tomogram
    start = timeit.default_timer()
    data = tomo.get_normalized(np.s_[:])
    stop = timeit.default_timer()
    print(f"Tomogram: {data.shape}, {data.dtype}, "
          f"[{data.min()}, {data.max()}]")
    print(f"Load time: {stop - start:5}s")

    # Save sinogram
    sino_index = 1080
    sino = data[:, sino_index, :]
    saveTiff(sino, f'./sinogram_{sino_index:04}', normalise=True)

    # Get mask
    start = timeit.default_timer()
    mask = np.load('../stripesmasksand.npz')['stripesmask']
    stop = timeit.default_timer()
    print(f"Mask: {mask.shape}, {mask.dtype}, [{mask.min()}, {mask.max()}]")
    print(f"Load time: {stop - start:5}s")
    filename = f'./mask_{sino_index:04}'
    saveTiff(mask[:, sino_index, :], filename, normalise=True)

    # Split data into windows and create input/target pairs
    createPariedWindows(data, mask, num_windows=5)
