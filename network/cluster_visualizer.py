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

    def get_patch(self, index, patch_num, sub_dir):
        """Get a single patch of a sinogram. If the patch_num given does not
        exist, an array of all zeroes will be returned.
        Parameters:
            index : int
                Index of the sinogram to retrieve the patch from.
            patch_num : int
                Number of the patch to get from the given sinogram.
            sub_dir : str
                Sub-directory to retrieve patches from.
                Must be either 'clean' or 'stripe'.
        Returns:
            np.ndarray
                The patch. 2D array with shape `self.patch_size`.
        """
        path = os.path.join(self.root, sub_dir, f'{index:04}_w{patch_num:02}')
        if os.path.exists(path+'.tif'):
            return loadTiff(path, normalise=False)
        else:
            return np.zeros(self.patch_size, dtype=np.uint16)

    def get_model_patch(self, index, patch_num):
        """Get the output of a model on a given patch.
        Parameters:
            index : int
                Index of the sinogram to retrieve the patch from.
            patch_num : int
                Number of the patch to get from the given sinogram.
        Returns:
              np.ndarray
                The model output on the given patch.
        """
        clean = self.get_patch(index, patch_num, 'clean')
        stripe = self.get_patch(index, patch_num, 'stripe')
        mask = np.abs(clean - stripe).astype(np.bool_, copy=False)
        mask = toTensor(mask).unsqueeze(0).type(torch.bool)
        stripe = toTensor(rescale(stripe, a=-1, b=1, imin=0, imax=65535)).unsqueeze(0)
        stripe[mask] = 0
        model_out = self.model(stripe)
        model_patch = stripe + mask * model_out
        return toNumpy(model_patch)

    def get_sinogram(self, index, sub_dir):
        """Get a full sinogram by combining its patches. If no patch is found
        for a part of the sinogram, that part will be set to 0.
        Parameters:
            index : int
                Index of the sinogram to return.
            sub_dir : str
                Sub-directory to retrieve sinograms from.
                Must be either 'clean' or 'stripe'.
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
            full_sino[current_idx] = self.get_patch(index, p, sub_dir)
        return full_sino

    def get_model_sinogram(self, index):
        """Get model output of a full sinogram by combining its patches.
        Parameters:
            index : int
                Index of the sinogram to run the model on.
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
            model_patch = self.get_model_patch(index, p)
            model_patch = rescale(model_patch, a=0, b=65535, imin=-1, imax=1)
            full_sino[current_idx] = model_patch
        return full_sino

    def plot_clean(self, index, show=True):
        """Plot a clean sinogram.
        Parameters:
            index : int
                Index of the sinogram to plot.
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The clean sinogram.
        """
        clean = self.get_sinogram(index, 'clean')
        plt.imshow(clean, cmap='gray', vmin=0, vmax=65535)
        plt.title(f"Clean {index}")
        if show:
            plt.show()
        return clean

    def plot_stripe(self, index, show=True):
        """Plot a stripe sinogram.
        Parameters:
            index : int
                Index of the sinogram to plot.
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The stripe sinogram.
        """
        stripe = self.get_sinogram(index, 'stripe')
        plt.imshow(stripe, cmap='gray', vmin=0, vmax=65535)
        plt.title(f"Stripe {index}")
        if show:
            plt.show()
        return stripe

    def plot_pair(self, index):
        """Plot a pair of clean & stripe sinograms.
        Parameters:
            index : int
                Index of the sinograms to plot.
        Returns:
             Tuple[np.ndarray, np.ndarray]
                The clean & stripe sinograms.
        """
        plt.subplot(121)
        clean = self.plot_clean(index, show=False)
        plt.subplot(122)
        stripe = self.plot_stripe(index, show=True)
        return clean, stripe

    def plot_model_sinogram(self, index, show=True):
        """Plot the output of the model on a sinogram.
        Parameters:
            index : int
                Index of the sinogram to run the model on.
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The output of the model on the given sinogram.
        """
        model_sino = self.get_model_sinogram(index)
        plt.imshow(model_sino, cmap='gray', vmin=0, vmax=65535)
        plt.title(f"Model Output {index}")
        if show:
            plt.show()
        return model_sino

    def plot_clean_recon(self, index, show=True):
        """Plot a reconstruction of a clean sinogram.
        Parameters:
            index : int
                Index of the sinogram to reconstruct and plot.
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The clean reconstruction.
        """
        clean = self.get_sinogram(index, 'clean')
        recon = reconstruct(rescale(clean))
        plt.imshow(recon, cmap='gray', vmin=-0.1, vmax=0.2)
        plt.title(f"Clean Reconstruction {index}")
        if show:
            plt.show()
        return recon

    def plot_stripe_recon(self, index, show=True):
        """Plot a reconstruction of a stripe sinogram.
        Parameters:
            index : int
                Index of the sinogram to reconstruct and plot.
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The stripe reconstruction.
        """
        stripe = self.get_sinogram(index, 'stripe')
        recon = reconstruct(rescale(stripe))
        plt.imshow(recon, cmap='gray', vmin=-0.05, vmax=0.1)
        plt.title(f"Stripe Reconstruction {index}")
        if show:
            plt.show()
        return recon

    def plot_pair_recon(self, index):
        """Plot a pair of clean & stripe reconstructions.
        Parameters:
            index : int
                Index of the sinograms to reconstruct and plot.
        Returns:
             Tuple[np.ndarray, np.ndarray]
                The clean & stripe reconstructions.
        """
        plt.subplot(121)
        clean = self.plot_clean_recon(index, show=False)
        plt.subplot(122)
        stripe = self.plot_stripe_recon(index, show=True)
        return clean, stripe

    def plot_model_recon(self, index, show=True):
        """Plot the reconstruction of the output of the model on a sinogram.
        Parameters:
            index : int
                Index of the sinogram to run the model on & reconstruct.
            show : bool
                Whether the plot should be displayed on screen. Default is True
        Returns:
             np.ndarray
                The reconstruction of the output of the model on the given
                sinogram.
        """
        model_sino = self.get_model_sinogram(index)
        recon = reconstruct(model_sino)
        plt.imshow(recon, cmap='gray', vmin=-0.1, vmax=0.2)
        plt.title(f"Model Output Reconstruction {index}")
        if show:
            plt.show()
        return model_sino

    def plot_all(self, index):
        """Plot the images from every stage of the process:
            Clean, Stripe, Model Output & each of their reconstructions.
        Parameters:
            index : int
                The index of the sinogram to plot.
        """
        plt.subplot(231)
        self.plot_clean(index, show=False)
        plt.subplot(232)
        self.plot_stripe(index, show=False)
        plt.subplot(233)
        self.plot_model_sinogram(index, show=False)
        plt.subplot(234)
        self.plot_clean_recon(index, show=False)
        plt.subplot(235)
        self.plot_stripe_recon(index, show=False)
        plt.subplot(236)
        self.plot_model_recon(index, show=True)


if __name__ == '__main__':
    root = '/dls/science/users/iug27979/NoStripesNet'
    model = ClusterUNet()
    checkpoint = torch.load(f'{root}/pretrained_models/cluster_1_lr0.0001.tar',
                            map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['gen_state_dict'])
    v = PatchVisualizer(f'{root}/data', model)

    sino_idx = 1089
    v.plot_all(sino_idx)
    v.plot_all(1666)
    v.plot_all(900)
