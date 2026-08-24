import pytest

from scripts.cosref_core import effective_parameters


def test_effective_parameter_mapping():
    a_eff, b_eff, theta_eff = effective_parameters(0.25, -0.10, 0.05, beta=5.0)
    assert a_eff == pytest.approx(0.05)
    assert b_eff == pytest.approx(-0.02)
    assert theta_eff == pytest.approx(-0.01)


def test_effective_parameter_mapping_requires_positive_beta():
    with pytest.raises(ValueError, match="beta must be positive"):
        effective_parameters(1.0, 1.0, 1.0, beta=0.0)
