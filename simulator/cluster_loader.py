import sys
import timeit
import os
import h5py as h5
from mpi4py import MPI
import numpy as np
import multiprocessing
from utils.data_io import saveTiff
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
            new_item = slice(item.start + lo,
                             item.stop + lo,
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
        print(f"{new_item=}")
        return self.data[new_item]


if __name__ == '__main__':
    tomo = TomoH5('/dls/i12/data/2022/nt33730-1/rawdata/119617.nxs')
    print(f"Data Shape: {tomo.shape}")
    if tomo.contains_flats():
        print(f"Flats: {tomo.flats.shape}")
    if tomo.contains_darks():
        print(f"Darks: {tomo.darks.shape}")

    # Save projection image
    proj_index = 900
    start = timeit.default_timer()
    proj = tomo[proj_index, :, :]
    stop = timeit.default_timer()
    print(f"Projection: {proj.shape}, {proj.dtype}, "
          f"[{proj.min()}, {proj.max()}]")
    print(f"Load time: {stop - start:5}s")
    filename = f'./projection_{proj_index:04}'
    saveTiff(proj, filename, normalise=False)

    # Save sinogram image
    sino_index = 1080
    start = timeit.default_timer()
    sino = tomo[:, sino_index, :]
    stop = timeit.default_timer()
    print(f"Sinogram: {sino.shape}, {sino.dtype}, "
          f"[{sino.min()}, {sino.max()}]")
    print(f"Load time: {stop - start:5}s")
    filename = f'./sinogram_{sino_index:04}'
    saveTiff(sino, filename, normalise=False)

    # Load entire 3D tomogram
    start = timeit.default_timer()
    data = tomo[:]
    stop = timeit.default_timer()
    print(f"Tomogram: {data.shape}, {data.dtype}, "
          f"[{data.min()}, {data.max()}]")
    print(f"Load time: {stop - start:5}s")

    # Apply stripe detection
    start = timeit.default_timer()
    mask = detect_stripe_larix(data)
    stop = timeit.default_timer()
    print(f"Mask: {mask.shape}, {mask.dtype}, "
          f"[{mask.min()}, {mask.max()}]")
    print(f"Time: {stop - start:5}s")
    filename = f'./mask_{sino_index:04}'
    saveTiff(mask[:, sino_index, :], filename, normalise=True)
