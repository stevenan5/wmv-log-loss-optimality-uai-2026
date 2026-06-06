import numpy as np
from scipy.io import savemat
from scipy.special import rel_entr
from dataset import Dataset
from mv import MV
from ocds import OCDS
from bf_dual_solvers import BFDualSolverConditional, BFConditional


# theta_1, gamma_1 are the weights for the prediction
# which we want to know if it's better or not than
# theta_2, gamma_2.  b_diff is the difference between
# the true accs/class freqs and those same quantities estimated by
# the prediction gotten by using theta_1, gamma_1
def our_test(
    b_diff,
    theta_1,
    gamma_1,
    theta_2,
    gamma_2,
    cond_pred_1,
    cond_pred_2,
    n_pts,
    preds_per_rule,
    scalar_b_diff=True,
):
    if scalar_b_diff:
        lhs = b_diff * np.ones(theta_1.size + gamma_1.size)
    lhs = np.linalg.norm(b_diff * preds_per_rule / n_pts, ord=2)
    num = rel_entr(cond_pred_1, cond_pred_2).sum() / n_pts
    den = np.linalg.norm(theta_1 - theta_2, ord=2) + np.linalg.norm(
        gamma_1 - gamma_2, ord=2
    )
    rhs = num / den

    return lhs <= rhs


def our_test_exact(
    b_star,
    b_est,
    theta_1,
    gamma_1,
    theta_2,
    gamma_2,
    cond_pred_1,
    cond_pred_2,
    n_pts,
    preds_per_rule,
):
    lhs = (
        (preds_per_rule * (b_star[: theta_1.size] - b_est[: theta_1.size]))
        @ (theta_2 - theta_1)
        / n_pts
    )
    lhs += (b_star[-gamma_1.size :] - b_est[-gamma_1.size :]) @ (gamma_2 - gamma_1)
    rhs = rel_entr(cond_pred_1, cond_pred_2).sum() / n_pts

    return lhs <= rhs


def an_dasgupta_test(
    eps,
    theta_star,
    gamma_star,
    train_labels,
    ds_labels,
    opt_labels,
    n_pts,
    eps_is_scalar=False,
):
    num = (
        rel_entr(train_labels, ds_labels) - rel_entr(train_labels, opt_labels)
    ).sum() / n_pts
    den = 2 * (np.linalg.norm(theta_star, ord=1) + np.linalg.norm(gamma_star, ord=1))
    if eps_is_scalar:
        lhs = eps
    else:
        lhs = np.linalg.norm(eps, ord=np.inf)

    rhs = num / den
    return lhs <= rhs, rhs


def compute_approx_uncert(opt_labels, our_labels, n_pts):
    return rel_entr(opt_labels, our_labels).sum() / n_pts


def construct_bf_problem(dataset):
    bf = BFConditional(
        dataset.n_rules,
        dataset.n_classes,
        np.zeros(dataset.n_rules),
        np.zeros(dataset.n_classes),
        rule_pred_constraints=dataset.train_preds,
    )
    true_accs, true_class_freqs = bf.compute_induced_quantities_from_labeling(
        dataset.train_labels_flat
    )
    bf.set_accs_class_freqs(true_accs, true_class_freqs)
    return bf


def construct_bf_dual_solver(bf, verbose=False):
    bf_dual_solver = BFDualSolverConditional(
        bf,
        bf.true_accs,
        bf.true_class_freqs,
        np.zeros((bf.n_rules, 2)),
        np.zeros((bf.n_classes, 2)),
        verbose=verbose,
    )
    return bf_dual_solver


def compute_dual_solution(bf_dual_solver, accs_error_bars, class_freqs_error_bars):
    bf_dual_solver.update_error_bars(accs_error_bars, class_freqs_error_bars)
    bf_dual_solver.solve()
    return bf_dual_solver.acc_weights, bf_dual_solver.class_freqs_weights


def compute_optimal_approximator(dataset, bf_dual_solver):
    theta_star, gamma_star = compute_dual_solution(
        bf_dual_solver, np.zeros((dataset.n_rules, 2)), np.zeros((dataset.n_classes, 2))
    )
    g_star = bf_dual_solver.bf.compute_prediction(theta_star, gamma_star, None)
    return g_star, theta_star, gamma_star


if __name__ == "__main__":
    datsets = [
        "agnews",
        "cdr",
        "chemprot",
        "commercial",
        "imdb",
        "semeval",
        "sms",
        "tennis",
        "trec",
        "yelp",
        "youtube",
    ]
    dataset_name = datsets[10]
    print(dataset_name)
    dataset = Dataset(dataset_name)
    eps_list = [0.01, 0.05, 0.1, 0.2, 0.3]

    res_dic = {
        "dataset": dataset_name,
        "eps_list": eps_list,
        "bf_better_than_ocds": [],
        "bf_better_than_mv": [],
        "our_test_bf_vs_ocds": [],
        "our_test_bf_vs_mv": [],
        "our_test_bf_vs_ocds_exact": [],
        "our_test_bf_vs_mv_exact": [],
        "an_dasg_test_ocds": [],
        "our_test_bf_vs_ocds_first_false_neg_eps": -1,
        "our_test_bf_vs_mv_first_false_neg_eps": -1,
        "an_dasg_bf_vs_ocds_first_false_neg_eps": -1,
        "an_dasg_test_rhs": -1,
    }

    bf = construct_bf_problem(dataset)
    bf_dual_solver = construct_bf_dual_solver(bf)
    best_approx, best_theta, best_gamma = compute_optimal_approximator(
        dataset, bf_dual_solver
    )

    n_rules = dataset.n_rules
    n_classes = dataset.n_classes
    n_points = dataset.n_points
    true_accs = bf.true_accs
    true_class_freqs = bf.true_class_freqs
    preds_per_rule = bf.preds_per_rule

    ocds = OCDS(n_rules, n_classes, true_accs, true_class_freqs, dataset.train_preds)
    mv = MV(n_rules, n_classes, true_accs, true_class_freqs, dataset.train_preds)

    ocds_oracle_pred = ocds.true_quantity_cond_dist
    # before running EM, the weights are the weights using the true quantities
    # as the true dist is computed in the OCDS constructor
    ocds_oracle_theta = ocds.em_rule_weights
    ocds_oracle_gamma = ocds.em_class_freqs_weights
    ocds_em_pred = ocds.em_alg()

    ocds_em_theta = ocds.em_rule_weights
    ocds_em_gamma = ocds.em_class_freqs_weights

    mv_pred = mv.compute_cond_dist()
    mv_theta = np.ones(n_rules)
    mv_gamma = np.ones(n_classes)

    # false negative detection, higher epsilon is better
    our_test_ocds_fn_detected = False
    our_test_mv_fn_detected = False
    an_dasg_ocds_fn_detected = False

    for eps in eps_list:
        curr_bf_theta, curr_bf_gamma = compute_dual_solution(
            bf_dual_solver, eps * np.ones((n_rules, 2)), eps * np.ones((n_classes, 2))
        )
        curr_bf_pred = bf_dual_solver.bf.compute_prediction(
            curr_bf_theta, curr_bf_gamma, None
        )
        bf_approx_uncert = compute_approx_uncert(best_approx, curr_bf_pred, n_points)
        # better_than_ocds = (
        #     compute_approx_uncert(best_approx, ocds_em_pred) > bf_approx_uncert
        # )
        better_than_ocds = (
            compute_approx_uncert(best_approx, ocds_oracle_pred, n_points)
            > bf_approx_uncert
        )
        better_than_mv = (
            compute_approx_uncert(best_approx, mv_pred, n_points) > bf_approx_uncert
        )
        ind_accs, ind_cf = bf.compute_induced_quantities_from_labeling(curr_bf_pred)
        our_test_ocds_exact = our_test_exact(
            np.concatenate((true_accs, true_class_freqs)),
            np.concatenate((ind_accs, ind_cf)),
            curr_bf_theta,
            curr_bf_gamma,
            ocds_oracle_theta,
            ocds_oracle_gamma,
            curr_bf_pred,
            ocds_oracle_pred,
            n_points,
            preds_per_rule,
        )
        our_test_mv_exact = our_test_exact(
            np.concatenate((true_accs, true_class_freqs)),
            np.concatenate((ind_accs, ind_cf)),
            curr_bf_theta,
            curr_bf_gamma,
            mv_theta,
            mv_gamma,
            curr_bf_pred,
            mv_pred,
            n_points,
            preds_per_rule,
        )
        our_test_ocds = our_test(
            eps,
            curr_bf_theta,
            curr_bf_gamma,
            ocds_oracle_theta,
            ocds_oracle_gamma,
            curr_bf_pred,
            ocds_oracle_pred,
            n_points,
            preds_per_rule,
            scalar_b_diff=True,
        )
        our_test_mv = our_test(
            eps,
            curr_bf_theta,
            curr_bf_gamma,
            mv_theta,
            mv_gamma,
            curr_bf_pred,
            mv_pred,
            n_points,
            preds_per_rule,
            scalar_b_diff=True,
        )
        an_dasgp_test, an_dasg_rhs = an_dasgupta_test(
            eps,
            best_theta,
            best_gamma,
            dataset.train_labels_flat,
            ocds_em_pred,
            best_approx,
            n_points,
            eps_is_scalar=True,
        )

        res_dic["bf_better_than_ocds"].append(better_than_ocds)
        res_dic["bf_better_than_mv"].append(better_than_mv)
        res_dic["our_test_bf_vs_ocds"].append(our_test_ocds)
        res_dic["our_test_bf_vs_mv"].append(our_test_mv)
        res_dic["our_test_bf_vs_ocds_exact"].append(our_test_ocds_exact)
        res_dic["our_test_bf_vs_mv_exact"].append(our_test_mv_exact)
        res_dic["an_dasg_test_ocds"].append(an_dasgp_test)
        res_dic["an_dasg_test_rhs"] = an_dasg_rhs

        # check for first false positive
        if not our_test_ocds_fn_detected and (better_than_ocds and not our_test_ocds):
            res_dic["our_test_bf_vs_ocds_first_false_neg_eps"] = eps
            our_test_ocds_fn_detected = True
        if not our_test_mv_fn_detected and (better_than_mv and not our_test_mv):
            res_dic["our_test_bf_vs_mv_first_false_neg_eps"] = (eps,)
            our_test_mv_fn_detected = True
        if not an_dasg_ocds_fn_detected and (better_than_ocds and not an_dasgp_test):
            res_dic["an_dasg_bf_vs_ocds_first_false_neg_eps"] = (eps,)
            an_dasg_ocds_fn_detected = True

        # check that our exact method is correct
        if np.all(
            res_dic["bf_better_than_ocds"] == res_dic["our_test_bf_vs_ocds_exact"]
        ):
            print("Exact test matches actual comparison for OCDS")
        if np.all(res_dic["bf_better_than_mv"] == res_dic["our_test_bf_vs_mv_exact"]):
            print("Exact test matches actual comparison for MV")

    output_fname = "./results/" + dataset_name + "_weight_comparison.mat"
    print(res_dic)
    savemat(output_fname, res_dic)
