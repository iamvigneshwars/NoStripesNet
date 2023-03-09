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
    weights = STRIPES_DETECT(data, size=250, radius=3)
    middle = timeit.default_timer()
    print("Merging stripe weights...")
    mask = STRIPES_MERGE(weights,
                         threshold=0.63,
                         min_stripe_length=600,
                         min_stripe_depth=30,
                         min_stripe_width=22,
                         sensitivity_perc=85.0)
    stop = timeit.default_timer()
    print(f"Times: "
          f"\tSTRIPES_DETECT:  {middle - start:5}s\n"
          f"\tSTRIPES_MERGE :  {stop - middle:5}s\n"
          f"\tTotal         :  {stop - start:5}s")
    filename = f'./mask_1080'
    saveTiff(mask[:, 1080, :], filename, normalise=True)
    return mask


def create_patches(data, patch_size):
    """Split a 2D array into a number of smaller 2D patches.
    Parameters:
        data : np.ndarray
            Data to be split into patches. If data does not evenly fit into
            the size of patch specified, it will be cropped.
        patch_size : Tuple[int, int]
            Size of patches. Must have form:
                (patch_height, patch_width)
    Returns:
        np.ndarray
            Array containing patches. Has shape:
                (num_patches, patch_height, patch_width)
    """
    # Check that data can be evenly split into patches of size `patch_size`
    remainder = np.mod(data.shape, patch_size)
    if remainder[0] != 0:
        # If patch height doesn't evenly go into image height, crop image
        if remainder[0] % 2 == 0:
            # If remainder is even, crop evenly on bottom & top
            data = data[remainder[0]//2:-(remainder[0]//2)]
        else:
            # Otherwise, crop one more from top
            data = data[remainder[0]//2:-(remainder[0]//2) - 1]
    if remainder[1] != 0:
        # If patch width doesn't evenly go into image width, crop image
        if remainder[1] % 2 == 0:
            # If remainder is even, crop evenly on left & right
            data = data[:, remainder[1]//2:-(remainder[1]//2)]
        else:
            # Otherwise, crop one more from right
            data = data[:, remainder[1]//2:-(remainder[1]//2) - 1]
    # First, split into patches by width
    num_patches_w = data.shape[1] // patch_size[1]
    patches_w = np.split(data, num_patches_w, axis=1)
    # Then, split into patches by height
    num_patches_h = data.shape[0] // patch_size[0]
    patches_h = [np.split(p, num_patches_h, axis=0) for p in patches_w]
    # Finally, combine into one array
    num_patches = num_patches_h * num_patches_w
    patches = np.asarray(patches_h).reshape(num_patches, *patch_size)
    return patches


def createPariedWindows(data, mask, patch_size):
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
        patch_size : Tuple[int, int]
            Patch size to split sinograms into. If sinogram shape does not go
            evenly into patch size, the sinogram will be cropped.
    """
    # Swap axes so sinograms are in axis 0
    # i.e. data has shape (detector Y, angles, detector X)
    data = np.swapaxes(data, 0, 1)
    mask = np.swapaxes(mask, 0, 1)
    # Loop through each sinogram
    for s in range(data.shape[0]):
        if s % 100 == 0:
            print(f"Processing sinogram {s:04}...", end=' ', flush=True)
        # Normalise sinogram
        sino = rescale(data[s], b=65535).astype(np.uint16)
        # Add synthetic stripes to sinogram
        # currently done with TomoPhantom, other methods could be used
        stripe = add_stripes(sino, percentage=2, maxthickness=2,
                             intensity_thresh=0.2,
                             stripe_type='full',
                             variability=0)
        # Clip back to original range
        stripe = np.clip(stripe, sino.min(), sino.max())
        # Split sinogram, mask, & stripe into windows
        sino_windows = create_patches(sino, patch_size)
        mask_windows = create_patches(mask[s], patch_size)
        stripe_windows = create_patches(stripe, patch_size)
        # Loop through each window
        for w in range(len(sino_windows)):
            # if sinogram doesn't contain stripes, create input/target pair
            if mask_windows[w].sum() == 0:
                # Save 'clean' target
                clean = sino_windows[w]
                filename = f'data/clean/{s:04}_w{w:02}'
                saveTiff(clean, filename, normalise=False)
                # Save 'stripe' input
                filename = f'data/stripe/{s:04}_w{w:02}'
                saveTiff(stripe_windows[w], filename, normalise=False)
            else:
                # Otherwise, save to different directory as real artifact
                stripe = sino_windows[w]
                filename = f'data/real_artifacts/{s:04}_w{w:02}'
                saveTiff(stripe, filename, normalise=False)
                # Save mask as well
                mask_w = mask_windows[w].astype(np.bool_)
                filename = f'data/real_artifacts/mask_{s:04}_w{w:02}'
                np.save(filename, mask_w)
        if s % 100 == 0:
            print("Done.")


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

    # Get mask
    start = timeit.default_timer()
    mask = np.load('../stripesmasksand.npz')['stripesmask']
    stop = timeit.default_timer()
    print(f"Mask: {mask.shape}, {mask.dtype}, [{mask.min()}, {mask.max()}]")
    print(f"Load time: {stop - start:5}s")

    # Split data into windows and create input/target pairs
    createPariedWindows(data, mask, patch_size=(1801, 256))
