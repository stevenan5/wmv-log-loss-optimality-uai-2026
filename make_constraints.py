# make constraint matrices where every pattern is present
import numpy as np
import itertools as it
from scipy.sparse import coo_array
from scipy.special import comb
from util import ind


def construct_pred_constraints(p, k, use_specialists=False):
    # number of patterns and predictions
    if use_specialists:
        tau = (k + 1) ** p - 1
        n_preds = 0
        for pp in range(1, p + 1):
            n_preds += pp * (k**pp) * comb(p, pp)
        n_preds = int(n_preds)
        n_preds_per_rule = n_preds // p
    else:
        tau = k**p
        n_preds_per_rule = tau
        n_preds = p * n_preds_per_rule
    tauk = k * tau

    # include an extra class, denoted k, to represent abstentions
    if use_specialists:
        classes = np.arange(k + 1)
    else:
        classes = np.arange(k)

    preds_sparse = list(it.product(classes, repeat=p))
    preds_sparse = [list(ele) for ele in preds_sparse]

    data = np.ones(n_preds)
    row = np.zeros((p, n_preds_per_rule))
    col = np.zeros((p, n_preds_per_rule))

    for j in range(p):
        row[j, :] = j * np.ones(n_preds_per_rule)
        col_ind = 0
        for t in range(tau):
            # skip if there is an abstention
            if use_specialists and preds_sparse[t][j] == k:
                continue
            else:
                # when not using specialists, col_ind == t
                col[j, col_ind] = ind(k, t, preds_sparse[t][j])
                col_ind += 1

    row = row.flatten()
    col = col.flatten()

    rule_constraints = coo_array(
        (data, (row, col)),
        shape=(p, tauk),
    )

    # also return list of patterns and number of patterns
    return rule_constraints, preds_sparse, tau


def construct_class_freq_constraints(k, n_patterns):
    tauk = k * n_patterns
    data = np.ones(tauk)
    row = np.zeros((k, n_patterns))
    col = np.zeros((k, n_patterns))

    for ell in range(k):
        row[ell, :] = ell * np.ones(n_patterns)
        for t in range(n_patterns):
            col[ell, t] = ind(k, t, ell)

    row = row.flatten()
    col = col.flatten()
    class_freq_constraints = coo_array(
        (data, (row, col)),
        shape=(k, tauk),
    )

    return class_freq_constraints


def construct_pattern_constraints(k, n_patterns):
    tauk = k * n_patterns
    data = np.ones(tauk)
    row_pattern = np.zeros(tauk)
    col_pattern = np.arange(tauk)

    for t in range(n_patterns):
        for ell in range(k):
            row_pattern[ind(k, t, ell)] = t

    pattern_constraints = coo_array(
        (data[:tauk], (row_pattern, col_pattern)), shape=(n_patterns, tauk)
    )

    return pattern_constraints
