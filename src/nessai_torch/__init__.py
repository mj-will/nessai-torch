"""nessai-torch

Implementation of nessai in PyTorch.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass


from .sampler import Sampler

__all__ = [
    "Sampler",
]
