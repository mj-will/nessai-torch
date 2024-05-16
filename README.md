# nessai-torch

Implementation of nessai: nested sampling with artificial intelligence in PyTorch.

## Installation

`nessai-torch` can be install using `pip`:

```
pip install nessai-torch
```

We recommend install PyTorch first to ensure the version is compatible with
your system.

## Basic usage

`nessai-torch` has a different API to `nessai`, the user must define a
log-likelihood function and a prior-transform function instead of a model
object. It also has a reduced feature set compared to standard `nessai`.
The basic usage is shown below, for a more complete example, see the
`examples` directory.

```python
from nessai_torch.sampler import Sampler


sampler = Sampler(
	log_likelihood=log_likelihood_fn,
	prior_transform=prior_transform_fn,
	dims=dims,  # Number of dimensions
)

# Run the sampler
sampler.run()
```

Note that both the `log_likelihood` and `prior_transform` must be vectorized.

## Citing

If you use nessai-torch in your work please cite the DOI and the relevant papers:

```bibtext
@article{Williams_2021,
	doi = {10.1103/physrevd.103.103006},
	url = {https://doi.org/10.1103%2Fphysrevd.103.103006},
	year = 2021,
	month = {may},
	publisher = {American Physical Society ({APS})},
	volume = {103},
	number = {10},
	author = {Michael J. Williams and John Veitch and Chris Messenger},
	title = {Nested sampling with normalizing flows for gravitational-wave inference},
	journal = {Physical Review D}
}

@article{Williams_2023,
	doi = {10.1088/2632-2153/acd5aa},
	url = {https://doi.org/10.1088%2F2632-2153%2Facd5aa},
	year = 2023,
	month = {jul},
	publisher = {{IOP} Publishing},
	volume = {4},
	number = {3},
	pages = {035011},
	author = {Michael J. Williams and John Veitch and Chris Messenger},
	title = {Importance nested sampling with normalising flows},
	journal = {Machine Learning: Science and Technology}
}
```
