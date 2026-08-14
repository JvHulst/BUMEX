"""
Check the regularized transition-probability solver.

The solution is verified three ways: against a general-purpose nonlinear solver
from SciPy, against the closed-form water-filling solution that the problem
reduces to at lambda = 0, and against the KKT conditions of the problem itself.
Feasibility is checked on every instance, and the batched and single-instance
entry points are checked to agree.

    python tests/test_fast_regularized.py
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from fast_regularized import solve, solve_batch, XI


def objective(p, p_data, V, lambda_reg, opt_type):
    """Value of the regularized objective at p."""
    p = np.maximum(np.asarray(p, dtype=float), XI)
    if opt_type == 'min':
        observed = p_data > 0
        kl = np.sum(p_data[observed] * np.log(p_data[observed] / p[observed]))
        return float(np.dot(p, V) + lambda_reg * kl)
    return float(np.dot(p, V) + lambda_reg * np.sum(p_data * np.log(p)))


def scipy_reference(lower, upper, p_data, V, lambda_reg, opt_type):
    """Optimum from a general-purpose solver, started from the box centre."""
    lower = np.maximum(lower, XI)
    sign = 1.0 if opt_type == 'min' else -1.0
    start = np.clip(np.full_like(V, 1.0 / len(V)), lower, upper)
    start = start / start.sum()
    result = minimize(lambda p: sign * objective(p, p_data, V, lambda_reg, opt_type),
                      np.clip(start, lower, upper),
                      method='SLSQP',
                      bounds=list(zip(lower, upper)),
                      constraints=[{'type': 'eq', 'fun': lambda p: p.sum() - 1.0}],
                      options={'maxiter': 500, 'ftol': 1e-12})
    return result.x


def water_filling(lower, upper, V, opt_type):
    """Closed-form optimum at lambda = 0: fill in order of the next-state value."""
    p = np.maximum(lower, XI).copy()
    order = np.argsort(V) if opt_type == 'min' else np.argsort(V)[::-1]
    for i in order:
        if p.sum() >= 1:
            break
        p[i] += min(1 - p.sum(), upper[i] - p[i])
    return p


def violation(p, lower, upper):
    """Worst absolute constraint violation over box, simplex and positivity."""
    lower = np.maximum(lower, XI)
    return max(np.max(np.maximum(lower - p, 0.0)),
               np.max(np.maximum(p - upper, 0.0)),
               abs(p.sum() - 1.0))


def kkt_residual(p, lower, upper, p_data, V, lambda_reg, opt_type):
    """
    Largest violation of stationarity on the coordinates that are off their bounds.

    Away from the box, the gradient of the Lagrangian vanishes, so
    V_i - lambda * p_data_i / p_i is the same for every such coordinate. Its
    spread is therefore a certificate that is independent of how p was computed.
    """
    lower = np.maximum(lower, XI)
    interior = (p > lower + 1e-9) & (p < upper - 1e-9)
    if interior.sum() < 2:
        return 0.0
    sign = 1.0 if opt_type == 'min' else -1.0
    gradient = sign * V - lambda_reg * p_data / p
    return float(np.ptp(gradient[interior]))


def random_instance(rng, n, zero_fraction=0.0):
    """A feasible instance resembling those seen during a Q-bound update."""
    # Scale with n so the instance stays feasible: sum(lower) <= 1 <= sum(upper).
    while True:
        lower = rng.uniform(0, 0.6 / n, n)
        upper = np.minimum(lower + rng.uniform(0.5 / n, 2.5 / n, n), 1.0)
        if lower.sum() <= 1.0 <= upper.sum():
            break
    counts = rng.integers(0, 50, n).astype(float)
    if zero_fraction:
        counts[rng.random(n) < zero_fraction] = 0.0
    p_data = counts / counts.sum() if counts.sum() > 0 else np.full(n, 1 / n)
    V = rng.uniform(-60, 40, n)
    return lower, upper, p_data, V


def test_against_scipy(rng):
    """The direct solution is at least as good as a general-purpose solver."""
    worst_gap = worst_kkt = worst_violation = 0.0
    trials = 0
    for n in (2, 3, 5, 9, 20):
        for zero_fraction in (0.0, 0.4):
            for lambda_reg in (0.0, 0.05, 1.0, 25.0):
                for _ in range(15):
                    lower, upper, p_data, V = random_instance(rng, n, zero_fraction)
                    for opt_type in ('min', 'max'):
                        p = solve(lower, upper, p_data, V, lambda_reg, opt_type)
                        p_ref = scipy_reference(lower, upper, p_data, V, lambda_reg, opt_type)

                        f, f_ref = (objective(p, p_data, V, lambda_reg, opt_type),
                                    objective(p_ref, p_data, V, lambda_reg, opt_type))
                        # 'min' should be no larger than the reference, 'max' no smaller.
                        gap = (f - f_ref) if opt_type == 'min' else (f_ref - f)
                        # Only count the reference where it is itself feasible.
                        if violation(p_ref, lower, upper) < 1e-7:
                            worst_gap = max(worst_gap, gap / max(1.0, abs(f_ref)))
                        worst_violation = max(worst_violation, violation(p, lower, upper))
                        worst_kkt = max(worst_kkt, kkt_residual(p, lower, upper, p_data,
                                                                V, lambda_reg, opt_type))
                        trials += 1

    print(f'  {trials} instances against SciPy SLSQP')
    print(f'  worst relative objective gap (positive is worse): {worst_gap:.3e}')
    print(f'  worst constraint violation:                       {worst_violation:.3e}')
    print(f'  worst KKT stationarity residual:                  {worst_kkt:.3e}')
    assert worst_gap < 1e-6, 'direct solution is worse than the reference'
    assert worst_violation < 1e-9, 'direct solution is infeasible'
    assert worst_kkt < 1e-5, 'direct solution does not satisfy stationarity'


def test_unregularized(rng):
    """At lambda = 0 the solution matches the closed form."""
    worst = 0.0
    for n in (2, 3, 5, 9, 20):
        for _ in range(40):
            lower, upper, p_data, V = random_instance(rng, n)
            for opt_type in ('min', 'max'):
                p = solve(lower, upper, p_data, V, 0.0, opt_type)
                p_ref = water_filling(lower, upper, V, opt_type)
                worst = max(worst, abs(objective(p, p_data, V, 0.0, opt_type)
                                       - objective(p_ref, p_data, V, 0.0, opt_type)))
    print(f'  worst objective difference from water-filling:    {worst:.3e}')
    assert worst < 1e-9, 'lambda = 0 does not reproduce the closed form'


def test_batch_matches_single(rng):
    """Rows solved in a batch match the same instances solved one at a time."""
    m, width = 500, 9
    lengths = rng.integers(2, width + 1, m)
    lower = np.zeros((m, width))
    upper = np.zeros((m, width))
    p_data = np.zeros((m, width))
    V = np.zeros((m, width))
    mask = np.zeros((m, width), dtype=bool)
    for row, k in enumerate(lengths):
        lo, up, pd, v = random_instance(rng, k, zero_fraction=0.3)
        mask[row, :k] = True
        lower[row, :k], upper[row, :k], p_data[row, :k], V[row, :k] = lo, up, pd, v
    lam = rng.choice([0.0, 0.05, 1.0, 25.0], m)

    worst = 0.0
    for opt_type in ('min', 'max'):
        batched = solve_batch(lower, upper, p_data, V * mask, lam, mask, opt_type)
        for row, k in enumerate(lengths):
            single = solve(lower[row, :k], upper[row, :k], p_data[row, :k],
                           V[row, :k], lam[row], opt_type)
            worst = max(worst, np.max(np.abs(batched[row, :k] - single)))
    print(f'  worst batched-vs-single difference:               {worst:.3e}')
    assert worst < 1e-12, 'batched and single-instance solutions disagree'


def test_infeasible_is_reported():
    """Bounds that cannot meet the simplex constraint raise rather than return."""
    for lower, upper in ((np.full(4, 0.4), np.full(4, 0.9)),
                         (np.zeros(4), np.full(4, 0.1))):
        try:
            solve(lower, upper, np.full(4, 0.25), np.arange(4.0), 1.0, 'min')
        except ValueError:
            continue
        raise AssertionError('infeasible bounds were accepted')
    print('  infeasible bounds rejected')


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for name, test in (('optimality', test_against_scipy),
                       ('unregularized limit', test_unregularized),
                       ('batch consistency', test_batch_matches_single)):
        print(f'{name}:')
        test(rng)
    print('infeasibility:')
    test_infeasible_is_reported()
    print('\nall checks passed')
