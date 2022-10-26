import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import utils
from tomobar.methodsDIR import RecToolsDIR

from datasets import *
from models.gans import *


def batch_reconstruct(batch, size, device='cpu'):
    total_angles = int(0.5 * np.pi * size)
    angles = np.linspace(0, 179.9, total_angles, dtype='float32')
    angles_rad = angles * (np.pi / 180.0)
    p = int(np.sqrt(2) * size)
    rectools = RecToolsDIR(DetectorsDimH=p,
                           DetectorsDimV=None,
                           CenterRotOffset=0.0,
                           AnglesVec=angles_rad,
                           ObjSize=size,
                           device_projector=device)
    assert (batch.shape[1] == 1)
    new_batch = torch.zeros((batch.shape[0], 1, size, size))
    for i in range(batch.shape[0]):  # is looping best idea? possible to vectorise?
        sinogram = batch[i].squeeze().numpy()
        sinogram = np.delete(sinogram, np.s_[p:], axis=-1)  # crop sinogram to correct size
        new_batch[i] = torch.from_numpy(rectools.FBP(sinogram)).unsqueeze(0)
    return new_batch


class BaseGANVisualizer:
    def __init__(self, model, dataloader, size):
        self.model = model
        self.gen = self.model.gen
        self.disc = self.model.disc
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        self.dataset = dataloader.dataset
        self.size = size

    def plot_losses(self):
        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.model.lossG_values, label="G")
        plt.plot(self.model.lossD_values, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_real_vs_fake_batch(self):
        """Function to plot a batch of real inputs, target outputs and generated outputs.
        Before running this function, at least one train or test pass must have been made."""
        # Plot the target outputs (i.e. clean sinograms)
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 3, 1)
        plt.axis("off")
        plt.title("Target Outputs")
        plt.imshow(np.transpose(utils.make_grid(self.model.realB.detach(), padding=5, normalize=True, nrow=4).cpu(),
                                (1, 2, 0)), cmap='gray')
        # Plot the real inputs (i.e. centre sinograms)
        plt.subplot(1, 3, 2)
        plt.axis("off")
        plt.title("Real Inputs")
        plt.imshow(np.transpose(utils.make_grid(self.model.realA.detach(), padding=5, normalize=True, nrow=4).cpu(),
                                (1, 2, 0)), cmap='gray')
        # Plot the fake outputs (i.e. generated sinograms)
        plt.subplot(1, 3, 3)
        plt.axis("off")
        plt.title("Generated Outputs")
        plt.imshow(np.transpose(utils.make_grid(self.model.fakeB.detach(), padding=5, normalize=True, nrow=4).cpu(),
                                (1, 2, 0)), cmap='gray')
        plt.show()

    def plot_real_vs_fake_recon(self):
        """Function to plot a batch of *reconstructed* real inputs, target outputs and generated outputs.
        Before running this function, at least one train or test pass must have been made."""
        # Reconstruct all
        input_recon = batch_reconstruct(self.model.realA.detach(), self.size)
        target_recon = batch_reconstruct(self.model.realB.detach(), self.size)
        fake_recon = batch_reconstruct(self.model.fakeB.detach(), self.size)
        # Plot clean vs centre vs generated reconstructions
        plt.figure(figsize=(8, 8))
        plt.subplot(131)
        plt.axis("off")
        plt.title("Targets")
        plt.imshow(np.transpose(utils.make_grid(target_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.subplot(132)
        plt.axis("off")
        plt.title("Inputs")
        plt.imshow(np.transpose(utils.make_grid(input_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.subplot(133)
        plt.axis("off")
        plt.title("Generated")
        plt.imshow(np.transpose(utils.make_grid(fake_recon, normalize=True, nrow=4, scale_each=True).cpu(), (1, 2, 0)),
                   cmap='gray')
        plt.show()


class PairedWindowGANVisualizer(BaseGANVisualizer):
    def __init__(self, model, dataloader, size):
        super().__init__(model, dataloader, size)

    def plot_real_vs_fake_batch(self):
        self.model.realA = self.dataset.combineWindows(self.model.realAs)
        self.model.realB = self.dataset.combineWindows(self.model.realBs)
        self.model.fakeB = self.dataset.combineWindows(self.model.fakeBs)
        super().plot_real_vs_fake_batch()

    def plot_real_vs_fake_recon(self):
        self.model.realA = self.dataset.combineWindows(self.model.realAs)
        self.model.realB = self.dataset.combineWindows(self.model.realBs)
        self.model.fakeB = self.dataset.combineWindows(self.model.fakeBs)
        super().plot_real_vs_fake_recon()