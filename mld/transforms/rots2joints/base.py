from typing import Optional

import torch
from torch import Tensor, nn
from pathlib import Path


class Rots2Joints(nn.Module):
    def __init__(self, path: Optional[str] = None,
                 normalization: bool = False,
                 eps: float = 1e-12,
                 **kwargs) -> None:
        if normalization and path is None:
            raise TypeError("You should provide a path if normalization is on.")

        super().__init__()
        self.normalization = normalization
        self.eps = eps

        if normalization:
            mean_path = Path(path) / "mean.pt"
            std_path = Path(path) / "std.pt"
            self.register_buffer('mean', torch.load(mean_path))
            self.register_buffer('std', torch.load(std_path))

    def normalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = (features - self.mean)/(self.std + self.eps)
        return features

    def unnormalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = features * self.std + self.mean
        return features
