from simulator.cluster_loader import TomoH5
import numpy as np
from skimage.transform import resize
from utils.data_io import saveTiff, loadTiff, rescale
from utils.stripe_detection import detect_stripe_larix
import os
from tomophantom.supp.artifacts import stripes as add_stripes


def resize_chunk(chunk, new_shape):
    return resize(chunk, new_shape, anti_aliasing=True)


def get_raw_data(tomogram):
    """Given a 3D tomogram (or subset), return the raw sinograms in
    it.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
    Returns:
        np.ndarray
            The raw sinograms in the given tomogram.
    """
    return resize_chunk(tomogram, (tomogram.shape[0], 402, 362))


def get_dynamic_data(tomogram, frame_angles=900):
    """Given a 3D tomogram (or subset), return the frames of the dynamic scan.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
        frame_angles : int
            Number of angles per frame of dynamic scan.
    Returns:
        np.ndarray
            4D array of shape (chunk idx, no. frames, sino Y, sino X).
    """
    num_frames = tomogram.shape[1] // (frame_angles * 2)
    out = np.ndarray((tomogram.shape[0], num_frames, 402, 362))
    for s in range(tomogram.shape[0]):
        for f in range(0, num_frames * 2, 2):
            frame = tomogram[s, f * frame_angles:(f + 1) * frame_angles, :]
            frame = resize_chunk(frame, (402, 362))
            out[s, f//2] = frame
    return out


def get_paired_data(tomogram, mask=None):
    """Given a 3D tomogram (or subset), return the pair of a clean sinogram
    and the same sinogram with stripes.
    Parameters:
        tomogram : np.ndarray
            3D tomographic data. Can be a subset of the full tomogram.
            Must have shape (detector Y, angles, detector X).
        mask : np.ndarray
            Mask indicating locations of stripes in the tomogram. If not
            provided, a mask will be generated (this can take multiple hours).
            Must have same shape as `tomogram`. Default is None.
    Returns:
        np.ndarray
            The clean/stripe pairs in the given tomogram.
    """
    # Resize
    tomogram = resize_chunk(tomogram, (tomogram.shape[0], 402, 362))
    out = np.ndarray(tomogram.shape[0], dtype=[('real_artifact', '?'),
                                               ('stripe', '<f8', (402, 362)),
                                               ('clean', '<f8', (402, 362))])
    # Generate mask if not given
    if mask is None:
        mask = detect_stripe_larix(tomogram, threshold=0.63)
    for s in range(tomogram.shape[0]):
        # if sinogram doesn't contain stripes, add stripes synthetically
        if np.sum(mask[s]) == 0:
            clean = tomogram[s]
            stripe_type = np.random.choice(['partial', 'full'])
            stripe = add_stripes(clean, percentage=1,
                                 maxthickness=2, intensity_thresh=0.2,
                                 stripe_type=stripe_type,
                                 variability=0.005)
            real_artifact = False
        else:
            # Otherwise, stripe is current sinogram
            clean = None  # Clean is not yet implemented
            stripe = tomogram[s]
            real_artifact = True
        out[s] = real_artifact, stripe, clean
    return out


def save_rescaled_sino(sino, imin, imax, path):
    """Helper function to rescale a sinogram and then save it.
    Also returns the min & max of the sinogram before rescaling.
    Parameters:
        sino : np.ndarray
            2D sinogram to save.
        imin : float
            Min value to rescale sinogram w.r.t.
        imax : float
            Max value to rescale sinogram w.r.t.
        path : str
            Path to save sinogram to.
    Returns:
        Tuple[float, float]
            Min and Max of sinogram before it was rescaled.
    """
    # Store sino min & max before rescaling
    smin, smax = sino.min(), sino.max()
    # Rescale sinogram w.r.t chunk min & max
    np.clip(sino, imin, imax, out=sino)
    sino = rescale(sino, b=65535, imin=imin, imax=imax)
    sino = sino.astype(np.uint16, copy=False)
    # Save sinogram to disk
    saveTiff(sino, path, normalise=False)
    return smin, smax


def save_chunk(chunk, root, mode, start=0, sample_num=0, shift_num=0):
    """Save a chunk of sinograms to disk, given a mode.
    Parameters:
        chunk: np.ndarray
            3D chunk containing sinograms to save.
            Must have shape (detector Y, angles, detector X).
        root : str
            Path to save sinograms to.
        mode : str
            Mode to save sinograms in.
            Must be one of ['raw', 'real', 'dynamic'].
        start : int
            Index of slice to start counting at.
        sample_num : int
            Number of sample. Used in filename when saving sinograms.
        shift_num : int
            Number of current shift. Used in filename when saving sinograms.
    Returns:
        Dict[str, Tuple[float, float]]
            Dictionary where key is path to tiff, and value is pair of min &
            max for each sinogram.
    """
    # Dictionary to store path & min/max of each sinogram
    minmax = {}
    # Store min & max of chunk for rescaling
    if mode == 'raw':
        chunk_min, chunk_max = chunk[20:-20].min(), chunk[20:-20].max()
        filepath = os.path.join(root, f'{sample_num:04}', f'shift{shift_num:02}')
        for s in range(chunk.shape[0]):
            filename = f'{sample_num:04}_shift{shift_num:02}_{start + s:04}'
            savepath = os.path.join(filepath, filename)
            sino = chunk[s]
            # Store path & min/max in dictionary
            minmax[savepath] = save_rescaled_sino(sino, chunk_min, chunk_max,
                                                  savepath)
    elif mode == 'dynamic':
        chunk_min, chunk_max = chunk[20:-20].min(), chunk[20:-20].max()
        for s in range(chunk.shape[0]):
            for f in range(chunk.shape[1]):
                frame = chunk[s, f]
                filename = f'{sample_num:04}_frame{f:02}_{start + s:04}'
                savepath = os.path.join(root, filename)
                minmax[savepath] = save_rescaled_sino(frame, chunk_min,
                                                      chunk_max, savepath)
    elif mode == 'real':
        chunk_min = np.nanmin(chunk[20:-20]['stripe'])
        chunk_max = np.nanmax(chunk[20:-20]['stripe'])
        basepath = os.path.join(root, f'{sample_num:04}')
        for s in range(chunk.shape[0]):
            real_artifact, stripe, clean = chunk[s]
            # Save to disk in correct directory based on whether artifact is
            # real or fake
            if real_artifact:
                filepath = os.path.join(basepath, 'real_artifacts')
            else:
                filepath = os.path.join(basepath, 'fake_artifacts')
            filename = f'{sample_num:04}_shift{shift_num:02}_{start + s:04}'
            stripe_path = os.path.join(filepath, 'stripe', filename)
            clean_path = os.path.join(filepath, 'clean', filename)
            # Save sino as tiff and store path & min/max in dictionary
            minmax[stripe_path] = save_rescaled_sino(stripe, chunk_min,
                                                     chunk_max, stripe_path)
            # clean may be None as it's not implemented for real artifacts
            if clean is not None:
                minmax[clean_path] = save_rescaled_sino(clean, chunk_min,
                                                        chunk_max, clean_path)
    return minmax


def chunk_generator(hdf_file, chunk_size):
    tomo = TomoH5(hdf_file)
    num_sinos = tomo.shape[1]
    num_chunks = int(np.ceil(num_sinos / chunk_size))
    for c in range(num_chunks):
        print(f"Loading chunk {c+1}/{num_chunks}...", end=' ', flush=True)
        chunk_slice = np.s_[:, c*chunk_size:(c+1)*chunk_size, :]
        chunk = tomo.get_normalized(chunk_slice)
        # Swap axes so sinograms are in dimension 0
        # i.e. (detector Y, angles, detector X)
        chunk = np.swapaxes(chunk, 0, 1)
        print(f"Done.")
        yield chunk


def reload_save(shape, minmax):
    full_tomo = np.ndarray(shape)
    for path, (lo, hi) in minmax.items():
        sino = loadTiff(path, normalise=True)
        sino = rescale(sino, a=lo, b=hi)
        sino_index = int(path[-4:])
        full_tomo[sino_index] = sino
    # Clip full tomo so it is not skewed by outliers
    full_tomo = np.clip(full_tomo,
                        full_tomo[20:-20].min(), full_tomo[20:-20].max())
    # Normalize again w.r.t. whole 3D tomogram
    full_tomo = rescale(full_tomo, a=0, b=65535)
    # Convert to uint16
    full_tomo = full_tomo.astype(np.uint16, copy=False)
    # Save each sinogram again
    for path in minmax.keys():
        sino_index = int(path[-4:])
        saveTiff(full_tomo[sino_index], path, normalise=False)


def get_data(mode, data, chunk_size, chunk_num, **kwargs):
    if mode == 'raw':
        return get_raw_data(data)
    elif mode == 'real':
        if 'mask' not in kwargs:
            raise ValueError("A mask should be given.")
        # Crop mask to correct size
        mask_idx = np.s_[chunk_num * chunk_size:(chunk_num+1) * chunk_size]
        mask = kwargs['mask'][mask_idx]
        return get_paired_data(data, mask=mask)
    elif mode == 'dynamic':
        if 'frame_angles' not in kwargs:
            raise ValueError("Angles per frame should be given.")
        return get_dynamic_data(data, frame_angles=kwargs['frame_angles'])
    else:
        raise ValueError(f"Mode must be one of ['raw', 'real', 'dynamic']. "
                         f"Instead got mode = '{mode}'.")


def generate_real_data(root, hdf_file, mode, chunk_size, **kwargs):
    rescale_dict = {}
    chunks = chunk_generator(hdf_file, chunk_size)
    num_chunks = 0
    for chunk in chunks:
        data = get_data(mode, chunk, chunk_size, num_chunks, **kwargs)
        current_idx = chunk_size * num_chunks
        chunk_dict = save_chunk(data, root, mode, start=current_idx)
        rescale_dict.update(chunk_dict)
        num_chunks += 1
    if num_chunks > 1:
        print(f"Re-loading & normalizing data w.r.t entire 3D sample...")
        if mode == 'raw':
            reload_save((num_chunks * chunk_size, 402, 362), rescale_dict)
        elif mode == 'real':
            # Create two dictionaries out of rescale_dict;
            # one for clean and one for stripe
            clean_dict, stripe_dict = {}, {}
            for path in rescale_dict.keys():
                if 'clean' in path:
                    clean_dict[path] = rescale_dict[path]
                elif 'stripe' in path:
                    stripe_dict[path] = rescale_dict[path]
            reload_save((num_chunks * chunk_size, 402, 362), stripe_dict)
            reload_save((num_chunks * chunk_size, 402, 362), clean_dict)
        elif mode == 'dynamic':
            reload_save((len(rescale_dict), 402, 362), rescale_dict)
