import glasflow
from glasflow import RealNVP
import torch


def get_realnvp(
    *,
    dims: int,
    n_neurons: int,
    n_transforms: int = 4,
    n_layers: int = 1,
    batch_norm_between_transforms: bool = True,
    compile: bool = False,
    **kwargs,
) -> glasflow.flows.base.Flow:
    flow = RealNVP(
        n_inputs=dims,
        n_transforms=n_transforms,
        n_blocks_per_transform=n_layers,
        n_neurons=n_neurons,
        batch_norm_between_transforms=batch_norm_between_transforms,
        **kwargs,
    )
    if compile:
        flow = torch.compile(flow)
    return flow
