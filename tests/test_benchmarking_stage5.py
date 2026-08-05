import numpy as np
from oulb.simulation import simulate_process
from oulb.benchmarking import fit_brownian_mle, fit_ou_mle, binary_metrics, adjusted_rand_index, cartesian_design

def test_brownian_fit_finite():
 t=np.linspace(0,10,100); r=simulate_process(t,[0.],model='brownian',drift=.2,sigma=.3,seed=1); f=fit_brownian_mle(t,r.states[:,0]); assert f['success'] and f['sigma']>0

def test_ou_fit_finite():
 t=np.linspace(0,20,200); r=simulate_process(t,[1.],model='ou',theta=.7,mu=.2,sigma=.3,seed=2); f=fit_ou_mle(t,r.states[:,0]); assert f['success'] and f['theta']>0

def test_binary_metrics():
 m=binary_metrics([1,0,1,0],[1,1,0,0]); assert m['tp']==1 and m['fp']==1

def test_ari_identity(): assert adjusted_rand_index([0,0,1,1],[1,1,0,0])==1.0

def test_design_size(): assert len(cartesian_design({'a':[1,2],'b':[3,4,5]}))==6


def test_ou_fit_is_finite_for_nearly_constant_series():
    import numpy as np
    from oulb.benchmarking import fit_ou_mle

    times = np.linspace(0.0, 5.0, 20)
    values = np.full(20, 0.25)
    values[-1] += 1e-8

    fit = fit_ou_mle(times, values)

    assert np.isfinite(fit["theta"])
    assert np.isfinite(fit["sigma"])
    assert np.isfinite(fit["loglik"])
    assert 1e-4 <= fit["theta"] <= 20.0
    assert 1e-6 <= fit["sigma"] <= 20.0


def test_standardized_ou_innovations_support_brownian_limit():
    import numpy as np
    from oulb.benchmarking import standardized_ou_innovations

    times = np.array([0.0, 0.5, 1.5, 3.0])
    values = np.array([0.0, 0.1, -0.1, 0.2])

    z = standardized_ou_innovations(
        times,
        values,
        mu=0.0,
        theta=0.0,
        sigma=0.25,
    )

    assert z.shape == (3,)
    assert np.all(np.isfinite(z))


def test_measurement_aware_ou_fit_returns_finite_values():
    import numpy as np
    from oulb.benchmarking import (
        fit_ou_mle_measurement_aware,
    )

    times = np.linspace(0.0, 5.0, 20)
    values = np.array([
        0.00, 0.10, -0.05, 0.08, 0.02,
        -0.04, 0.06, 0.01, -0.02, 0.05,
        0.00, -0.03, 0.04, 0.02, -0.01,
        0.03, -0.02, 0.01, 0.00, 0.02,
    ])

    fit = fit_ou_mle_measurement_aware(
        times,
        values,
        measurement_sd=0.1,
    )

    assert np.isfinite(fit["theta"])
    assert np.isfinite(fit["sigma"])
    assert np.isfinite(fit["loglik"])
    assert 1e-4 <= fit["theta"] <= 20.0
    assert 1e-6 <= fit["sigma"] <= 20.0


def test_measurement_aware_zero_noise_matches_naive_scale():
    import numpy as np
    from oulb.benchmarking import (
        fit_ou_mle,
        fit_ou_mle_measurement_aware,
    )

    times = np.linspace(0.0, 5.0, 30)
    values = np.sin(times) * 0.2

    naive = fit_ou_mle(times, values)
    aware = fit_ou_mle_measurement_aware(
        times,
        values,
        measurement_sd=0.0,
    )

    assert np.isclose(
        naive["theta"],
        aware["theta"],
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.isclose(
        naive["sigma"],
        aware["sigma"],
        rtol=1e-5,
        atol=1e-6,
    )
