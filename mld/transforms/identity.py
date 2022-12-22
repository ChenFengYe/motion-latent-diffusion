from typing import Optional
from torch import Tensor

from .base import Datastruct, dataclass, Transform


class IdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return IdentityDatastruct(**kwargs)

    def __repr__(self):
        return "IdentityTransform()"


@dataclass
class IdentityDatastruct(Datastruct):
    transforms: IdentityTransform

    features: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features"]

    def __len__(self):
        return len(self.rfeats)
