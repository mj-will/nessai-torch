import pytest


@pytest.mark.requires("bilby")
@pytest.mark.requires("numpy")
def test_bilby_integration(tmp_path):
    import bilby
    import numpy as np

    class SimpleGaussianLikelihood(bilby.Likelihood):
        def __init__(self):
            super().__init__(parameters={"x": None, "y": None})

        def log_likelihood(self):
            """Log-likelihood."""
            return -0.5 * (
                self.parameters["x"] ** 2.0 + self.parameters["y"] ** 2.0
            ) - np.log(2.0 * np.pi)

    priors = dict(
        x=bilby.core.prior.Uniform(-10, 10, "x"),
        y=bilby.core.prior.Uniform(-10, 10, "y"),
    )

    likelihood = SimpleGaussianLikelihood()

    outdir = tmp_path / "bilby_test"

    bilby.run_sampler(
        outdir=outdir,
        plot=False,
        likelihood=likelihood,
        priors=priors,
        sampler="nessai_torch",
        injection_parameters={"x": 0.0, "y": 0.0},
        seed=1234,
        nlive=50,
        tolerance=5.0,
    )
