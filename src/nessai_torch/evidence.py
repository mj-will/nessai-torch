from typing import Optional
import torch

from .utils.maths import log_integrate_log_trap, logsubexp
from .tensorlist import TensorList


class EvidenceIntegral:
    def __init__(self, *, nlive: int, device: torch.DeviceObjType):
        self.device = device
        self.nlive = nlive
        self.logt = torch.tensor(
            -1.0 / nlive, requires_grad=False, device=device
        )
        self.logz = -torch.tensor(
            torch.inf, requires_grad=False, device=device
        )
        self.logw = torch.tensor(0.0, requires_grad=False, device=device)
        self.logx = TensorList(device=device)
        self.logx.append(torch.tensor(0.0, requires_grad=False, device=device))
        self.logl = TensorList(device=device)
        self.logl.append(
            torch.tensor(-torch.inf, requires_grad=False, device=device)
        )
        self.info = torch.tensor(0.0, requires_grad=False, device=device)

    def update(self, logl: torch.Tensor, nlive: Optional[int] = None):
        """Update the integral"""
        logz_current = self.logz.clone()
        if nlive is not None:
            logt = torch.tensor(-1.0 / nlive, device=self.device)
        else:
            logt = self.logt
        logz = self.logw + torch.log1p(-torch.exp(self.logt)) + logl
        self.logz = torch.logaddexp(self.logz, logz)
        self.info = (
            torch.exp(logz - self.logz) * logl
            + torch.exp(logz_current - self.logz) * (self.info + logz_current)
            - self.logz
        )
        if torch.isnan(self.info):
            self.info = torch.tensor(
                0.0, requires_grad=False, device=self.device
            )
        self.logw += logt
        self.logx.append(self.logw.clone())
        self.logl.append(logl)

    @property
    def log_posterior_weights(self) -> torch.Tensor:
        """Compute the log-posterior weights."""
        logl = torch.cat((self.logl.data, self.logl[-1].view(1)), dim=0)
        logx = torch.cat(
            (
                self.logx.data,
                torch.tensor(-torch.inf, device=self.device).view(1),
            ),
            dim=0,
        )
        logz = log_integrate_log_trap(logl, logx)
        logw = logsubexp(logx[:-1], logx[1:])
        log_post_w = logl[1:-1] + logw[:-1] - logz
        return log_post_w

    @property
    def logz_error(self) -> torch.Tensor:
        return torch.sqrt(self.info / self.nlive)
