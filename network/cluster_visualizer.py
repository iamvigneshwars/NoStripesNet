import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from utils.data_io import loadTiff, rescale
from utils.tomography import reconstruct
from utils.misc import toTensor, toNumpy, plot_images
from network.models.cluster_models import ClusterUNet


class PatchVisualizer:
    """Class to visualize patches of a sinogram"""
    def __init__(self, root, model, full_sino_size=(1792, 2560),
                 patch_size=(256, 256)):
        """Parameters:
            root : str
                Path to root of dataset
            model : torch.nn.Module
                The model to process results from.
            full_sino_size : Tuple[int, int]
                Size of full sinograms; i.e. once patches have been combined.
                Default is (1792, 2560).
            patch_size : Tuple[int, int]
                Size of patches. Default is (256, 256).
        """
        self.root = root
        self.model = model
        self.size = full_sino_size
        self.patch_size = patch_size
        self.num_patches_h = self.size[0] // self.patch_size[0]
        self.num_patches_w = self.size[1] // self.patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

    def get_patch(self, index, patch_num, mode):
        """Get a single patch of a sinogram. If the patch_num given does not
        exist, an array of all zeroes will be returned.
        Parameters:
            index : int
                Index of the sinogram to retrieve the patch from.
            patch_num : int
                Number of the patch to get from the given sinogram.
            mode : str
                Mode in which to retrieve patches.
                Must be one of 'clean', 'stripe', or 'raw'.
        Returns:
            np.ndarray
                The patch. 2D array with shape `self.patch_size`.
        """
        if mode == 'raw':
            sub_dir = 'clean'
        else:
            sub_dir = mode
        path = os.path.join(self.root, sub_dir, f'{index:04}_w{patch_num:02}')
        if os.path.exists(path+'.tif'):
            return loadTiff(path, normalise=False)
        else:
            if mode == 'raw':
                path = os.path.join(self.root, 'real_artifacts',
                                    f'{index:04}_w{patch_num:02}')
                return loadTiff(path, normalise=False)
            else:
                return np.zeros(self.patch_size, dtype=np.uint16)

    def get_model_patch(self, index, patch_num, artifact_type):
        """Get the output of a model on a given patch.
        Parameters:
            index : int
                Index of the sinogram to retrieve the patch from.
            patch_num : int
                Number of the patch to get from the given sinogram.
            artifact_type : str
                Indicates the type of data the model should be ran on.
                'fake' = use fake artifacts and get mask from the absolute
                         difference between 'clean' and 'stripe'
                'real' = use real artifacts and get mask from stripe detection
                         algorithm. Assumes mask has already been saved to disk
        Returns:
              np.ndarray
                The model output on the given patch.
        """
        if artifact_type == 'fake':
            clean = self.get_patch(index, patch_num, 'clean')
            stripe = self.get_patch(index, patch_num, 'stripe')
            mask = np.abs(clean - stripe).astype(np.bool_, copy=False)
        elif artifact_type == 'real':
            stripe = self.get_patch(index, patch_num, 'raw')
            mask_file = os.path.join(self.root, 'real_artifacts',
                                     f'mask_{index:04}_w{patch_num:02}.npy')
            if os.path.exists(mask_file):
                mask = np.load(mask_file)
            else:
                # If no mask exists for this patch, then it doesn't contain a
                # stripe, so can be immediately returned as is.
                return stripe
        else:
            raise ValueError(f"Mode must be one of ['fake', 'real']. "
                             f"Instead got '{artifact_type}'.")
        mask = toTensor(mask).unsqueeze(0).type(torch.bool)
        stripe = toTensor(rescale(stripe, a=-1, b=1, imin=0,
                                  imax=65535)).unsqueeze(0)
        stripe[mask] = 0
        model_out = self.model(stripe)
        model_patch = stripe + mask * model_out
        return rescale(toNumpy(model_patch), a=0, b=65535, imin=-1,
                       imax=1).astype(np.uint16, copy=False)

    def get_sinogram(self, index, mode):
        """Get a full sinogram by combining its patches. If no patch is found
        for a part of the sinogram, that part will be set to 0.
        Parameters:
            index : int
                Index of the sinogram to return.
            mode : str
                Mode in which to retrieve patches.
                Must be one of 'clean', 'stripe', or 'raw'.
        Returns:
            np.ndarray
                The full sinogram.
        """
        full_sino = np.empty(self.size, dtype=np.uint16)
        for p in range(self.num_patches):
            current_idx = np.s_[
                          (p % self.num_patches_h) * self.patch_size[0]:
                          (p % self.num_patches_h + 1) * self.patch_size[0],
                          (p // self.num_patches_h) * self.patch_size[1]:
                          (p // self.num_patches_h + 1) * self.patch_size[1]]
            full_sino[current_idx] = self.get_patch(index, p, mode)
        return full_sino

    def get_reconstruction(self, index, mode):
        """Get a reconstruction of a full sinogram by combining its patches.
        If no patch is found for a part of the sinogram, that part will be set
        to 0 (this will cause errors in the reconstruction).
        Parameters:
            index : int
                Index of the sinogram to return.
            mode : str
                Mode in which to retrieve patches.
                Must be one of 'clean', 'stripe', or 'raw'.
        Returns:
            np.ndarray
                The reconstruction of the full sinogram.
        """
        sino = self.get_sinogram(index, mode)
        recon = reconstruct(rescale(sino))
        return recon

    def get_model_sinogram(self, index, artifact_type):
        """Get model output of a full sinogram by combining its patches.
        Parameters:
            index : int
                Index of the sinogram to run the model on.
            artifact_type : str
                Indicates the type of data the model should be ran on.
                'fake' = use fake artifacts and get mask from the absolute
                         difference between 'clean' and 'stripe'
                'real' = use real artifacts and get mask from stripe detection
                         algorithm. Assumes mask has already been saved to disk
        Returns:
            np.ndarray
                The output of the model on a full sinogram. Has type np.uint16.
        """
        full_sino = np.empty(self.size, dtype=np.uint16)
        for p in range(self.num_patches):
            current_idx = np.s_[
                          (p % self.num_patches_h) * self.patch_size[0]:
                          (p % self.num_patches_h + 1) * self.patch_size[0],
                          (p // self.num_patches_h) * self.patch_size[1]:
                          (p // self.num_patches_h + 1) * self.patch_size[1]]
            model_patch = self.get_model_patch(index, p, artifact_type)
            full_sino[current_idx] = model_patch
        return full_sino

    def get_model_reconstruction(self, index, artifact_type):
        """Get a reconstruction of model output of a full sinogram.
        Parameters:
            index : int
                Index of the sinogram to run the model on.
            artifact_type : str
                Indicates the type of data the model should be ran on.
                'fake' = use fake artifacts and get mask from the absolute
                         difference between 'clean' and 'stripe'
                'real' = use real artifacts and get mask from stripe detection
                         algorithm. Assumes mask has already been saved to disk
        Returns:
            np.ndarray
                The reconstruction of the output of the model on a full
                sinogram.
        """
        sino = self.get_model_sinogram(index, artifact_type)
        recon = reconstruct(rescale(sino))
        return recon

    def plot_sinogram(self, index, mode, show=True):
        """Plot a sinogram of a given mode.
        Parameters:
            index : int
                Index of the sinogram to plot.
            mode : str
                The type of sinogram to plot. Must be one of:
                'clean' ---------- sinogram with no artifacts and maybe missing
                                   patches
                'stripe' --------- sinogram with synthetic artifacts and maybe
                                   missing patches
                'real_artifacts' - sinogram with real artifacts and no missing
                                   patches
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The clean sinogram.
        """
        sino = self.get_sinogram(index, mode)
        plt.imshow(sino, cmap='gray', vmin=0, vmax=65535)
        plt.title(f"{mode.capitalize()} {index}")
        if show:
            plt.show()
        return sino

    def plot_reconstruction(self, index, mode, show=True):
        """Plot a reconstruction of a sinogram.
        Parameters:
            index : int
                Index of the sinogram to reconstruct.
            mode : str
                The type of sinogram to reconstruct. Must be one of:
                'clean' ---------- sinogram with no artifacts and maybe missing
                                   patches
                'stripe' --------- sinogram with synthetic artifacts and maybe
                                   missing patches
                'real_artifacts' - sinogram with real artifacts and no missing
                                   patches
            show : bool
                Whether the plot should be displayed on screen. Default is True
        """
        recon = self.get_reconstruction(index, mode)
        plt.imshow(recon, cmap='gray', vmin=-0.1, vmax=0.2)
        plt.title(f"{mode.capitalize()} Reconstruction {index}")
        if show:
            plt.show()

    def plot_model_sinogram(self, index, artifact_type, show=True):
        """Plot the output of the model on a sinogram.
        Parameters:
            index : int
                Index of the sinogram to run the model on.
            artifact_type : str
                Indicates the type of artifacts the model should be ran on.
                'fake' = use fake artifacts and get mask from the absolute
                         difference between 'clean' and 'stripe'
                'real' = use real artifacts and get mask from stripe detection
                         algorithm. Assumes mask has already been saved to disk
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The output of the model on the given sinogram.
        """
        model_sino = self.get_model_sinogram(index, artifact_type)
        plt.imshow(model_sino, cmap='gray', vmin=0, vmax=65535)
        plt.title(f"Model Output {index}")
        if show:
            plt.show()
        return model_sino

    def plot_model_reconstruction(self, index, artifact_type, show=True):
        """Plot reconstruction of the output of the model on a sinogram.
        Parameters:
            index : int
                Index of the sinogram to run the model on.
            artifact_type : str
                Indicates the type of artifacts the model should be ran on.
                'fake' = use fake artifacts and get mask from the absolute
                         difference between 'clean' and 'stripe'
                'real' = use real artifacts and get mask from stripe detection
                         algorithm. Assumes mask has already been saved to disk
            show : bool
                Whether the plot should be displayed on screen. Default is True
        """
        recon = self.get_model_reconstruction(index, artifact_type)
        plt.imshow(recon, cmap='gray', vmin=-0.1, vmax=0.2)
        plt.title(f"Model Output Reconstruction {index}")
        if show:
            plt.show()

    def plot_pair(self, index, recon=False):
        """Plot a pair of clean & stripe sinograms.
        Parameters:
            index : int
                Index of the sinograms to plot.
            recon : bool
                Whether the sinograms should be reconstructed as well.
        Returns:
             Tuple[np.ndarray, np.ndarray]
                The clean & stripe sinograms.
        """
        if recon:
            subplot_size = (2, 2)
        else:
            subplot_size = (1, 2)
        plt.subplot(*subplot_size, 1)
        clean = self.plot_sinogram(index, 'clean', show=False)
        plt.subplot(*subplot_size, 2)
        stripe = self.plot_sinogram(index, 'stripe', show=False)
        if recon:
            plt.subplot(*subplot_size, 3)
            self.plot_reconstruction(index, 'clean', show=False)
            plt.subplot(*subplot_size, 4)
            self.plot_reconstruction(index, 'stripe', show=False)
        plt.show()
        return clean, stripe

    def plot_all(self, index, recon=True):
        """Plot the images from every stage of the process:
            Clean, Stripe, Model Output & each of their reconstructions.
        Parameters:
            index : int
                The index of the sinogram to plot.
            recon : bool
                Whether reconstructions should be plotted below sinograms.
        """
        if recon:
            subplot_size = (2, 3)
        else:
            subplot_size = (1, 3)
        plt.subplot(*subplot_size, 1)
        self.plot_sinogram(index, 'clean', show=False)
        plt.subplot(*subplot_size, 2)
        self.plot_sinogram(index, 'stripe', show=False)
        plt.subplot(*subplot_size, 3)
        self.plot_model_sinogram(index, 'fake', show=False)
        if recon:
            plt.subplot(*subplot_size, 4)
            self.plot_reconstruction(index, 'clean', show=False)
            plt.subplot(*subplot_size, 5)
            self.plot_reconstruction(index, 'stripe', show=False)
            plt.clim(-0.01, 0.03)
            plt.subplot(*subplot_size, 6)
            self.plot_model_reconstruction(index, 'fake', show=False)
            plt.clim(-0.1, 0.15)
        fig = plt.gcf()
        plt.show()
        fig.set_size_inches((11, 8.5), forward=False)
        fig.savefig(f'./images/{sino_idx}_realArt_var0', dpi=500)

    def plot_all_raw(self, index, recon=True):
        if recon:
            subplot_size = (2, 2)
        else:
            subplot_size = (1, 2)
        plt.subplot(*subplot_size, 1)
        self.plot_sinogram(index, 'raw', show=False)
        plt.subplot(*subplot_size, 2)
        self.plot_model_sinogram(index, 'real', show=False)
        if recon:
            plt.subplot(*subplot_size, 3)
            self.plot_reconstruction(index, 'raw', show=False)
            plt.subplot(*subplot_size, 4)
            self.plot_model_reconstruction(index, 'real', show=False)
        fig = plt.gcf()
        plt.show()
        fig.set_size_inches((11, 8.5), forward=False)
        fig.savefig(f'./images/{sino_idx}_realArt_var0', dpi=500)


if __name__ == '__main__':
    root = '/dls/science/users/iug27979/NoStripesNet'
    model = ClusterUNet()
    checkpoint = torch.load(f'{root}/pretrained_models/cluster_1_variabilityZero.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['gen_state_dict'])
    v = PatchVisualizer(f'{root}/data', model)

    for sino_idx in [900, 1089, 1666]:
        v.plot_all(sino_idx, recon=True)
    for sino_idx in [852, 1080, 1200]:
        v.plot_all_raw(sino_idx, recon=True)
