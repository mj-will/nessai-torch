import math

import torch


@torch.jit.script
def logsubexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m = torch.max(a, b)
    return m + ((a - m).exp() - (b - m).exp()).log()


@torch.jit.script
def log_integrate_log_trap(
    logf: torch.Tensor, logx: torch.Tensor
) -> torch.Tensor:
    """
    Trapezoidal integration of given log(f). Returns log of the integral.

    Parameters
    ----------
    log_func : array_like
        Log values of the function to integrate over.
    log_support : array_like
        Log prior-volumes for each value.

    Returns
    -------
    float
        Log of the result of the integral.
    """
    logf_sum = torch.logaddexp(logf[:-1], logf[1:]) - math.log(2)
    log_dxs = logsubexp(logx[:-1], logx[1:])
    return torch.logsumexp(logf_sum + log_dxs, 0)
