import timeit
import numpy as np
from larix.methods.misc import STRIPES_MERGE

from simulator.cluster_loader import TomoH5
from utils.data_io import saveTiff
from utils.misc import plot_images


def test_merge_parameters(weights, parameter, parameter_range):
    """Generate and save a series of masks using different parameter values.
    Masks are all saved under a single numpy compressed '.npz' file.
    Masks are saved with the keyword:
        '<parameter>_<current_param_value>'
    where <current_param_value> is the value of `parameter` used to generate
    that mask.
    Parameters:
        weights: np.ndarray
            Stripe weights to generate masks from, output from STRIPES_DETECT
            function in Larix.
        parameter : str
            Name of parameter to alter for each mask generation.
        parameter_range : iterable
            Iterable of parameter values to use for each mask generation.
    """
    # Define default parameters
    parameters = dict(threshold=0.6,
                      min_stripe_length=600,
                      min_stripe_depth=30,
                      min_stripe_width=22,
                      sensitivity_perc=85.0)
    # Initialise empty dictionary of masks
    masks = {}
    # Loop through each parameter in range
    for param in parameter_range:
        # Set given parameter name to current parameter value
        parameters[parameter] = param
        # Calculate mask with current set of parameters
        masks[f'{parameter}_{param}'] = STRIPES_MERGE(weights, **parameters)
    # Save masks to numpy zip file
    np.savez_compressed('./test_merge_parameters.npz', **masks)
    return masks


def get_masks(pre_made=False, weights_file=None, masks_file=None):
    """Get a list of masks and the merge parameter range that created those
    masks.
    Parameters:
        pre_made : bool
            If True, will get a pre-made list of masks from disk, and will
                     infer parameter name and range from this list.
            If False, will ask user for parameter name and range, then will
                      create list of masks from these parameters.
        weights_file : str
            File of weights array, from Larix's STRIPES_DETECT method. Must be
            passed if `pre_made` is False.
        masks_file : str
            File of masks. Must be passed if `pre_made` is True.
    """
    if pre_made:
        if masks_file is None:
            raise AttributeError(
                "If `pre_made` is True, `masks_file` must be given.")
        # Load masks
        print("Loading masks from .npz file...")
        start = timeit.default_timer()
        masks = np.load(masks_file)
        # Get parameter name & range
        keys = masks.files
        param_name = keys[0].split('_')[0]
        param_range = [float(k.split('_')[1]) for k in keys]
        # Load each mask from zip file
        masks = [masks[k] for k in keys]
        print(f"Done in {timeit.default_timer() - start:.5f}s")
        return masks, param_name, param_range
    else:
        if weights_file is None:
            raise AttributeError(
                "If `pre_made` is False, `weights_file` must be given.")
        # Get weights
        print(f"Loading weights...")
        start = timeit.default_timer()
        weights = np.load(weights_file)['stripe_weights']
        print(f"Done in {timeit.default_timer() - start:.5f}s")
        print(f"Weights: {weights.shape}, {weights.dtype}, "
              f"[{weights.min()}, {weights.max()}]")
        # Get parameter name & range from user
        param_name = input("Enter name of parameter to test: ")
        param_range = input("Enter the values of the parameter you would like "
                            "to test, separated by commas: ")
        param_range = [float(p) for p in param_range.split(',')]
        # Test given parameter range
        masks = test_merge_parameters(weights, param_name, param_range)
        return masks.values(), param_name, param_range


if __name__ == '__main__':
    # Generate masks for a set of parameters
    # or load masks from disk if already generated
    pre_made = True
    weights_file = '../stripe_weights.npz'
    masks_file = './test_merge_parameters.npz'
    masks, param_name, param_range = get_masks(pre_made,
                                               weights_file,
                                               masks_file)

    # Load tomogram
    print("Loading tomo...")
    start = timeit.default_timer()
    tomo = TomoH5('/dls/i12/data/2022/nt33730-1/rawdata/119617.nxs')
    data = tomo.get_normalized(np.s_[:])
    print(f"Done in {timeit.default_timer() - start:.5f}s")

    # Let user choose a sinogram
    inpt = input("Enter sinogram index or quit (q): ")
    while inpt != 'q':
        try:
            sino_index = int(inpt)
            sino = data[:, sino_index, :]
            plot_images(sino, titles=[f"Sinogram {sino_index}"])
            if input("See masks for this sinogram? ([y], n)") != 'n':
                images = [sino] + [m[:, sino_index, :] for m in masks]
                titles = [f"Sinogram {sino_index}"] + \
                         [f"{param_name}={p}" for p in param_range]
                plot_images(*images, titles=titles, subplot_size=(3, 2))
        except (ValueError, IndexError):
            print("Please enter an integer in range [0, 2159]")
        inpt = input("Enter sinogram index or quit (q): ")
