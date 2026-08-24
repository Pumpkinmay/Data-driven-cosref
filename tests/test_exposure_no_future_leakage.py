import pytest

from scripts.build_digg_exposure_pilot import exposure_values


def test_current_and_future_adopters_do_not_enter_exposure_counts():
    # Node 2 adopted earlier; nodes 3 and 4 adopt in the current and future bins.
    # adopted_before is deliberately contaminated to test the defensive time check.
    result = exposure_values(
        node_id=1,
        adopted_before={2, 3, 4},
        adoption_bins={2: 0, 3: 1, 4: 2},
        time_bin=1,
        incoming={1: (2, 3, 4)},
        communities={1: 0, 2: 0, 3: 1, 4: 1},
    )
    m_in, m_out, degree, frac_in, frac_out, violations = result
    assert (m_in, m_out, degree) == (1, 0, 3)
    assert frac_in == pytest.approx(1 / 3)
    assert frac_out == 0
    assert violations == 2
