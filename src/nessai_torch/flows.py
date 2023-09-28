import glasflow
from glasflow import RealNVP


def get_realnvp(
    *,
    dims: int,
    n_neurons: int,
    n_transforms: int = 4,
    n_layers: int = 1,
    **kwargs,
) -> glasflow.flows.base.Flow:
    return RealNVP(
        n_inputs=dims,
        n_transforms=n_transforms,
        n_blocks_per_transform=n_layers,
        n_neurons=n_neurons,
        **kwargs,
    )
