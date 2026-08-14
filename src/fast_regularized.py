"""
Solver for the regularized transition-probability optimization behind the Q-bounds.

For one (state, action) pair with possible next states i, the lower Q-bound needs

    min_p  sum_i p_i V_i + lambda * sum_i p_data_i log(p_data_i / p_i)
    s.t.   lower <= p <= upper,  sum_i p_i = 1,

and the upper Q-bound the corresponding maximization.

At lambda = 0 the objective is linear, so the optimum raises each coordinate to
its upper bound in order of the next-state value until the mass reaches one.
`_water_fill` does that in closed form.

For lambda > 0, stationarity of the Lagrangian with multiplier nu on the simplex
constraint gives p_i = lambda * p_data_i / (V_i + nu). Imposing the box turns
this into

    p_i(nu) = clip( lambda * p_data_i / (V_i + nu),  lower_i,  upper_i ),

which is non-increasing in nu, so the nu satisfying sum_i p_i(nu) = 1 is found by
bisection in `_bisect`. The maximization has the same structure with denominator
(nu - V_i). Where the denominator is non-positive the objective is improved by
pushing that coordinate to its upper bound, which is also the limit of the
expression, so the mapping stays continuous and monotone.

Both routines take a whole value-iteration sweep at once: every array is (m, k),
one padded row per state-action pair. Rows go to whichever routine applies, so
the bisection runs only where regularization is active.
"""

import numpy as np

XI = 1e-8              # floor on p, keeping the logarithmic terms finite
BISECTION_ITERS = 80   # cap on the bisection; the bracket usually closes sooner
TOL_NU = 1e-14         # relative bracket width that counts as converged


def solve(lower_vec, upper_vec, p_data_vec, V_vec, lambda_reg, opt_type='min'):
    """
    Return the optimal transition distribution over the possible next states.

    Args:
        lower_vec, upper_vec: box bounds on the transition probabilities
        p_data_vec: empirical transition distribution
        V_vec: value of each possible next state
        lambda_reg: regularization weight; zero gives the unregularized problem
        opt_type: 'min' for the lower Q-bound, 'max' for the upper Q-bound
    """
    def as_row(vector):
        """Present a single instance as the one-row batch that solve_batch takes."""
        return np.asarray(vector, dtype=float).reshape(1, -1)

    mask = np.ones((1, len(V_vec)), dtype=bool)
    return solve_batch(as_row(lower_vec), as_row(upper_vec), as_row(p_data_vec),
                       as_row(V_vec), np.array([lambda_reg], dtype=float), mask,
                       opt_type)[0]


def solve_batch(lower, upper, p_data, V, lambda_reg, mask, opt_type='min'):
    """
    Solve one instance per row.

    Every array is (m, k), one padded row per state-action pair, with `mask`
    marking the real entries; `lambda_reg` is (m,). Padded columns are held at
    zero so they leave the simplex constraint untouched.

    Args:
        lower, upper: (m, k) box bounds on the transition probabilities
        p_data: (m, k) empirical transition distributions
        V: (m, k) value of each possible next state
        lambda_reg: (m,) regularization weight per instance
        mask: (m, k) boolean, True on the real entries of each row
        opt_type: 'min' for the lower Q-bound, 'max' for the upper Q-bound

    Raises:
        ValueError: if a row has no feasible p, i.e. sum(lower) > 1 or
            sum(upper) < 1.
    """
    if opt_type not in ('min', 'max'):
        raise ValueError(f'Unknown optimization type: {opt_type}')

    lower = np.where(mask, np.maximum(lower, XI), 0.0)
    upper = np.where(mask, upper, 0.0)
    p_data = np.where(mask, p_data, 0.0)
    V = np.where(mask, V, 0.0)
    lam = np.asarray(lambda_reg, dtype=float).ravel()

    # The box has to intersect the simplex. An infeasible row would otherwise
    # come back as a vector that does not sum to one, without any signal.
    lower_sum, upper_sum = lower.sum(axis=1), upper.sum(axis=1)
    infeasible = (lower_sum > 1.0 + 1e-9) | (upper_sum < 1.0 - 1e-9)
    if infeasible.any():
        row = int(np.argmax(infeasible))
        raise ValueError(
            f'infeasible probability bounds in row {row}: '
            f'sum(lower)={lower_sum[row]:.6g}, sum(upper)={upper_sum[row]:.6g}')

    p = _water_fill(lower, upper, V, mask, opt_type)

    # Regularization only changes the answer where lambda is non-zero and the
    # row leaves a choice; a single possible next state forces p = 1 on it.
    rows = np.flatnonzero((lam > 0) & (mask.sum(axis=1) > 1))
    if rows.size:
        width = int(np.flatnonzero(mask[rows].any(axis=0))[-1]) + 1
        columns = slice(0, width)
        p[rows, columns] = _bisect(lower[rows, columns], upper[rows, columns],
                                   lam[rows, None] * p_data[rows, columns],
                                   V[rows, columns], mask[rows, columns], opt_type)
    return p


def _water_fill(lower, upper, V, mask, opt_type):
    """
    Optimum of the unregularized problem.

    The objective is linear, so the mass left over above sum(lower) goes to the
    coordinates with the lowest next-state value first, or the highest for the
    maximization, until each is capped at its upper bound. Padded columns carry
    zero capacity and so take nothing.
    """
    key = np.where(mask, V if opt_type == 'min' else -V, np.inf)
    order = np.argsort(key, axis=1)
    capacity = np.take_along_axis(upper - lower, order, axis=1)
    budget = 1.0 - lower.sum(axis=1, keepdims=True)
    taken_earlier = np.cumsum(capacity, axis=1) - capacity
    fill = np.clip(budget - taken_earlier, 0.0, capacity)

    delta = np.empty_like(fill)
    np.put_along_axis(delta, order, fill, axis=1)
    return lower + delta


def _bisect(lower, upper, numerator, V, mask, opt_type):
    """
    Optimum of the regularized problem, one row per instance.

    `numerator` is lambda * p_data, the only place lambda enters. The multiplier
    nu is bracketed and then bisected until sum(p) = 1.
    """
    def p_of(nu):
        """
        The candidate p at multiplier nu, one row per instance.

        Padded columns have lower = upper = 0, so the clip pins them at zero.
        """
        denominator = (V + nu) if opt_type == 'min' else (nu - V)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = np.where(denominator > 0, numerator / denominator, np.inf)
        return np.clip(raw, lower, upper)

    # p_of is non-increasing in nu, so the bracket needs a low end whose mass
    # exceeds 1 and a high end whose mass falls below it. The low end must be
    # free to go below -min(V) (or max(V) for the maximization): coordinates
    # whose denominator has turned non-positive sit at their upper bound, and
    # feasibility gives sum(upper) >= 1, so a low enough nu always overshoots.
    largest = np.where(mask, V, -np.inf).max(axis=1, keepdims=True)
    smallest = np.where(mask, V, np.inf).min(axis=1, keepdims=True)
    centre = -smallest if opt_type == 'min' else largest
    span_initial = np.maximum(1.0, np.maximum(np.abs(V).max(axis=1, keepdims=True),
                                              numerator.max(axis=1, keepdims=True)))

    nu_low, span = centre.copy(), span_initial.copy()
    for _ in range(60):
        short = p_of(nu_low).sum(axis=1, keepdims=True) < 1.0
        if not short.any():
            break
        nu_low = np.where(short, nu_low - span, nu_low)
        span = np.where(short, span * 2.0, span)

    nu_high, span = centre + span_initial, span_initial.copy()
    for _ in range(60):
        over = p_of(nu_high).sum(axis=1, keepdims=True) > 1.0
        if not over.any():
            break
        span = np.where(over, span * 2.0, span)
        nu_high = np.where(over, centre + span, nu_high)

    for _ in range(BISECTION_ITERS):
        nu_mid = 0.5 * (nu_low + nu_high)
        heavy = p_of(nu_mid).sum(axis=1, keepdims=True) > 1.0
        nu_low = np.where(heavy, nu_mid, nu_low)
        nu_high = np.where(heavy, nu_high, nu_mid)
        if np.all(nu_high - nu_low <= TOL_NU * (1.0 + np.abs(nu_low))):
            break

    # A coordinate with no observed data makes p_of a step function of nu: it
    # jumps between its bounds as nu crosses the point where its denominator
    # vanishes. Bisection therefore stops on either side of the optimum, and the
    # unallocated mass belongs entirely to the coordinate at that jump. Once the
    # bracket is tight the two ends agree on every smoothly varying coordinate,
    # so interpolating between them to hit sum(p) = 1 places the remainder
    # exactly where it belongs without having to identify the jump.
    p_high, p_low = p_of(nu_high), p_of(nu_low)
    mass_high = p_high.sum(axis=1, keepdims=True)
    mass_low = p_low.sum(axis=1, keepdims=True)
    gap = mass_low - mass_high
    usable = gap > 1e-15
    t = np.where(usable, np.clip((1.0 - mass_high) / np.where(usable, gap, 1.0), 0.0, 1.0), 0.0)
    return np.clip(p_high + t * (p_low - p_high), lower, upper)
