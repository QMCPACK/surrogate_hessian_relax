from pytest import raises
from stalk.util.EffectiveVariance import EffectiveVariance
from stalk.util.util import match_to_tol


def test_EffectiveVariance():
    # Test nominal
    samples = 6
    errorbar = 5.0
    ev = EffectiveVariance(samples, errorbar)
    # Test that we recover original samples
    assert ev.get_samples(errorbar) == samples
    # Test that we recover original errorbar
    assert ev.get_errorbar(samples) == errorbar
    # Test that half the errorbar needs quadruple samples
    assert ev.get_samples(errorbar / 2) == 4 * samples
    # Test that twice the samples results in 1/sqrt(2) errorbar
    match_to_tol(ev.get_errorbar(2 * samples) == errorbar * 2**-0.5, 0.1)
    # Test that samples is always > 0
    assert ev.get_samples(1e10) == 1

    # Test degraded inits
    with raises(AssertionError):
        EffectiveVariance(0, errorbar)
    with raises(AssertionError):
        EffectiveVariance(samples, 0)
    with raises(TypeError):
        EffectiveVariance(samples)
    with raises(TypeError):
        EffectiveVariance()
# end def
