# KLIEP algorithm for sequential change point detection from the papers
# “Direct importance estimation for covariate shift adaptation” (Annals of the Institute of Statistical Mathematics, 2008)
# by M. Sugiyama, T. Suzuki, S. Nakajima, H. Kashima, P. von Bunau, and M. Kawanabe
# and
# "Change-point detection in time-series data by relative density-ratio estimation" (Neural Networks, 2013)
# by S. Liu, M. Yamada, N. Collier, and M. Sugiyama

import math

import cvxpy as cvx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def change_series(series: np.ndarray | pd.Series, type_change=None):
    if type_change == "log_simple":
        new_series = (np.log1p(np.roll(series, -1)) - np.log1p(series))[:-1]
    elif type_change == "lin":
        new_series = (np.roll(series, -1) - series)[:-1]
    else:
        new_series = np.log1p(abs(np.roll(series, -1) - series) / abs(series + 1))[:-1]

    return new_series


def kliep(X_te, X_re, sigma):
    # Test sample size
    n_te = X_te.shape[0]
    # Reference sample size
    n_re = X_re.shape[0]

    # Compute pairwise distances
    te_te_dist = pairwise_distances(X_te)
    re_te_dist = pairwise_distances(X_re, X_te)

    # Compute kernel matrices
    te_te_kernel = np.exp(-0.5 * (te_te_dist / sigma) ** 2)
    re_te_kernel = np.exp(-0.5 * (re_te_dist / sigma) ** 2)

    # Initialize a vector of coefficients
    theta = cvx.Variable(n_te)

    # Objective
    obj = cvx.Maximize(cvx.sum(cvx.log(te_te_kernel @ theta)))

    # Constraints
    constraints = [cvx.sum(re_te_kernel @ theta) == n_re, theta >= 0]

    # Problem
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver="SCS", max_iters=200, eps=1e-1)

    return obj.value


def compute_test_stat_kliep(X, window_size=10, sigma=0.1, threshold=math.inf):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)

    # More convenient notation
    b = window_size

    # Sample size
    n = X.shape[0]

    # Initialization
    T = np.zeros(n)
    stopping_time = -1

    for t in range(2 * b + 1, n):
        # Test sample
        X_te = X[t - b : t]
        # Reference sample
        X_re = X[t - 2 * b : t - b]

        T[t] = kliep(X_te, X_re, sigma)

        if T[t] > threshold:
            stopping_time = t
            break

    # Array of test statistics
    if stopping_time != -1:
        T = T[: stopping_time + 1]

    return T, stopping_time


def train_kliep(
    data_val, sigma, z, min_diff, window_size, change_points_val, verbose=False
):
    S_kliep_train = np.empty(0)

    # Initialization of the list of detected change points
    change_points_kliep = []

    # Initialization of the delays array and
    # the false alarms counter
    delays_kliep = np.empty(0)
    current_change_point_ind = 0
    false_alarms_kliep = 0

    # Initialization
    step_kliep = 0
    new_step_kliep = 0

    while new_step_kliep >= 0:
        # Run the procedure until the moment
        # it reports a change point occurrence
        new_S_kliep, new_step_kliep = compute_test_stat_kliep(
            data_val[step_kliep + 1 :],
            window_size=window_size,
            sigma=sigma,
            threshold=z,
        )

        S_kliep_train = np.append(S_kliep_train, new_S_kliep)

        step_kliep += new_step_kliep

        if new_step_kliep > 0:
            change_points_kliep += [int(step_kliep)]

            if verbose:
                print("Detected change point:", step_kliep)
            if current_change_point_ind >= len(change_points_val) or (
                change_points_val[current_change_point_ind] - step_kliep > min_diff
            ):
                if verbose:
                    print("False Alarm")
                false_alarms_kliep += 1
            else:
                skipped_cp = 0
                while (
                    current_change_point_ind < len(change_points_val)
                    and change_points_val[current_change_point_ind] <= step_kliep
                ):
                    if skipped_cp > 0:
                        delays_kliep = np.append(
                            delays_kliep,
                            np.array(
                                [
                                    change_points_val[current_change_point_ind]
                                    - change_points_val[current_change_point_ind - 1]
                                ]
                            ),
                            axis=0,
                        )
                    skipped_cp += 1

                    current_change_point_ind += 1
                if (
                    current_change_point_ind < len(change_points_val)
                    and change_points_val[current_change_point_ind] - step_kliep
                    <= min_diff
                ):
                    delays_kliep = np.append(delays_kliep, np.array([0.0]), axis=0)

                    if skipped_cp > 0:
                        delays_kliep = np.append(
                            delays_kliep,
                            np.array(
                                [
                                    change_points_val[current_change_point_ind]
                                    - change_points_val[current_change_point_ind - 1]
                                ]
                            ),
                            axis=0,
                        )

                    current_change_point_ind += 1
                    continue

                delays_kliep = np.append(
                    delays_kliep,
                    np.array(
                        [step_kliep - change_points_val[current_change_point_ind - 1]]
                    ),
                    axis=0,
                )

    while current_change_point_ind < len(change_points_val):
        if current_change_point_ind == len(change_points_val) - 1:
            delays_kliep = np.append(
                delays_kliep,
                np.array([len(data_val) - change_points_val[current_change_point_ind]]),
                axis=0,
            )
            break

        delays_kliep = np.append(
            delays_kliep,
            np.array(
                [
                    change_points_val[current_change_point_ind + 1]
                    - change_points_val[current_change_point_ind]
                ]
            ),
            axis=0,
        )
        current_change_point_ind += 1
    print(
        "KLIEP, threshold =",
        z,
        ". sigma =",
        sigma,
        ". Number of false alarms:",
        false_alarms_kliep,
        "; average delay:",
        np.mean(delays_kliep),
        "±",
        np.std(delays_kliep),
    )

    return S_kliep_train, delays_kliep


def perform_kliep(data_val, sigma, z, window_size, verbose=False):
    """
    Performs KLIEP (Kullback-Leibler importance estimation procedure) changepoint detection
    with passed hyperparameters.
    Returns statistic values and found changepoints.
    """
    S_kliep_train = np.empty(0)

    # Initialization of the list of detected change points
    change_points_kliep = []

    # Initialization
    step_kliep = 0
    new_step_kliep = 0

    while new_step_kliep >= 0:
        # Run the procedure until the moment
        # it reports a change point occurrence
        new_S_kliep, new_step_kliep = compute_test_stat_kliep(
            data_val[step_kliep + 1 :],
            window_size=window_size,
            sigma=sigma,
            threshold=z,
        )

        S_kliep_train = np.append(S_kliep_train, new_S_kliep)

        step_kliep += new_step_kliep

        if new_step_kliep > 0:
            change_points_kliep += [int(step_kliep)]
            if verbose:
                print("Detected change point:", step_kliep)

    return S_kliep_train, change_points_kliep
