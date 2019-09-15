import logging

import torch
import numpy as np
import scipy.io as sio
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    def __init__(self, mat_path, patch_size, transform=None):
        super().__init__()

        logger.info(f"loading .mat from {mat_path}...")
        data = sio.loadmat(mat_path)

        self._A = data["a"]
        self._B = data["b"]
        
        h, w = self._A.shape[2], self._A.shape[3]
        if patch_size is not None:
            xc = w // 2
            yc = h // 2
            lx = xc - patch_size // 2
            ux = xc - patch_size // 2 + patch_size
            ly = yc - patch_size // 2
            uy = yc - patch_size // 2 + patch_size

            self._A = self._A[..., ly:uy, lx:ux]
            self._B = self._B[..., ly:uy, lx:ux]

        logger.info(f"shape of the patches is {self._A.shape}")

        self._transform = transform

    def __getitem__(self, index):
        pa = torch.from_numpy(self._A[index]).float() / 255.0
        pb = torch.from_numpy(self._B[index]).float() / 255.0

        if self._transform is not None:
            pa = self._transform(pa)
            pb = self._transform(pb)

        return pa, pb

    def __len__(self):
        return self._A.shape[0]


if __name__ == "__main__":
    logger.info("running test")
    dset = PatchDataset(
        "/data/linguangzhang/patches91/carpet00_c91_p91_n512000_rand_ra.mat"
    )
