from .base import Proposal

import torch


class SphericalProposal(Proposal):
    """Proposal that draws samples from within a sphere with radius given by
    the worst point.

    Parameters
    ----------
    dims
        The number of dimensions.
    device
        The device to use for new samples.
    """

    populated: bool = True

    def __init__(self, *, dims: int, device: torch.DeviceObjType) -> None:
        super().__init__(dims=dims, device=device)
        self.origin = 0.5 * torch.ones(self.dims, device=device)

    def draw(self, sample: torch.tensor, n=1) -> torch.tensor:
        max_radius = torch.sqrt((torch.sum((sample - self.origin) ** 2)))
        z = torch.randn(n, self.dims, device=self.device)
        z = z / torch.sqrt(torch.sum(z**2, dim=1, keepdim=True))
        r = max_radius * (
            torch.rand(n, 1, device=self.device) ** (1 / self.dims)
        )
        x = r * z + self.origin
        return torch.squeeze(x), None
