import itertools
import numpy as np
import math
import matplotlib.pyplot as plt
import gmpy2
from os.path import join
from typing import List
from scipy.io import loadmat
from weight_optimality import (
    WeightOptimality,
)
from weighting_strategy import OneCoinDawidSkeneWeights, MajorityVoteWeights

from weight_optimality import Polytope, POLYTOPE_FILE_NAMES
from util import LINESTYLE_DICT

plt.rcParams.update({"font.size": 18})


def read_data(
    pattern_counter: WeightOptimality,
    class_freq_dist,
    n_rules,
    n_steps,
    command,
    polytopes: List[Polytope],
    reshape_results_to_matrix=False,
):
    if n_rules != 2 and reshape_results_to_matrix:
        raise ValueError("Cannot reshape counts/volumes to matrix unless n_rules is 2.")

    data = loadmat(pattern_counter.bookkeeping_fname)
    # this is when all permutations of accuracies are considered,
    # i.e. with the itertools.product call below
    # block_size = n_steps**n_rules
    block_size = math.comb(n_steps + n_rules - 1, n_rules)
    found_enum = False
    file_enum = 0
    rounding_epsilon = data["rounding_epsilon"][0]
    appropriateness_epsilon = float(
        gmpy2.mpfr(gmpy2.mpq(data["appropriateness_epsilon"][0]))
    )
    polytope_dims_dict = {polytope: [] for polytope in polytopes}
    # cast to string b/c that's how it's stored.
    class_freq_dist = [str(class_freq) for class_freq in class_freq_dist]

    pos_to_all_ratio_res = []
    opt_to_pos_ratio_res = []
    if command == "count":
        prob_type = "mass"
    elif command == "integrate":
        prob_type = "density"
    else:
        raise ValueError("command ", command, "is not an accepted argument.")

    polytope_suffix = POLYTOPE_FILE_NAMES[Polytope.POSSIBLE_PATTERNS_VREP]

    while not found_enum:
        read_class_freqs = data.get("true_class_freqs_" + str(file_enum))
        if read_class_freqs is None:
            raise ValueError(
                "Could not match requested class frequencies ",
                class_freq_dist,
                " in bookkeeping file. Maybe replot=True when",
                " it should be False?",
            )
        read_class_freqs = np.array(read_class_freqs).squeeze()
        if np.all(read_class_freqs == class_freq_dist):
            # test to see that a desired key exists
            # when both "count" and "integrate" are used as
            # command, we need to distinguish the file enums
            if (
                data.get(
                    "possible_to_all_"
                    + prob_type
                    + "_ratio_"
                    + polytope_suffix
                    + "_"
                    + str(file_enum)
                )
                is not None
            ):
                break
        file_enum += block_size

    for i in range(file_enum, file_enum + block_size):
        pta = np.ndarray.item(
            np.array(
                data[
                    "possible_to_all_"
                    + prob_type
                    + "_ratio_"
                    + polytope_suffix
                    + "_"
                    + str(i)
                ]
            )
        )
        pta_float = float(gmpy2.mpq(pta))
        otp = np.ndarray.item(
            np.array(
                data[
                    "opt_to_possible_"
                    + prob_type
                    + "_ratio_"
                    + polytope_suffix
                    + "_"
                    + str(i)
                ]
            )
        )
        otp_float = float(gmpy2.mpq(otp))
        pos_to_all_ratio_res.append(pta_float)
        opt_to_pos_ratio_res.append(otp_float)

        for polytope in polytopes:
            pname = POLYTOPE_FILE_NAMES[polytope]
            polytope_dims_dict[polytope].append(
                int(
                    np.ndarray.item(
                        np.array(
                            data["polytope_dimension_" + pname + "_" + str(file_enum)]
                        )
                    )
                )
            )

    polytope_avg_dim = {np.mean(polytope_dims_dict[polytope]) for polytope in polytopes}

    # for when block_size == n_steps ** n_rules
    if reshape_results_to_matrix:
        # pos_to_all_ratio_res = np.array(pos_to_all_ratio_res).reshape((-1, block_size))
        # opt_to_pos_ratio_res = np.array(opt_to_pos_ratio_res).reshape((-1, block_size))
        pta_mat = np.zeros((n_steps, n_steps))
        otp_mat = np.zeros((n_steps, n_steps))
        tril_inds = tril_indices_minor_diag(n_steps)
        pta_mat[tril_inds] = pos_to_all_ratio_res
        otp_mat[tril_inds] = opt_to_pos_ratio_res
        # reflect the matrix so we have a full matrix of results
        pta_diag = minor_diag(pta_mat.copy())
        otp_diag = minor_diag(otp_mat.copy())
        # subtract the minor diagonal because it appears twice in the sum
        pta_mat += np.flip(pta_mat) - pta_diag
        otp_mat += np.flip(otp_mat) - otp_diag
        otp_res = otp_mat
        pta_res = pta_mat
    else:
        otp_res = opt_to_pos_ratio_res
        pta_res = pos_to_all_ratio_res
    return otp_res, pta_res, rounding_epsilon, appropriateness_epsilon, polytope_avg_dim


# returns the indices for a lower triangular matrix
# where the entries are under the minor rather than
# major diagonal (which is what np.tril_indices returns)
def tril_indices_minor_diag(n_steps, offset=0):
    row_inds = []
    col_inds = []
    for i in range(n_steps):
        for j in range(offset + i, n_steps):
            row_inds.append(i)
            col_inds.append(j)

    return (row_inds, col_inds)


def minor_diag(array):
    if array.shape[0] != array.shape[1] or array.ndim != 2:
        raise ValueError("minor_diag argument must be square")
    n_steps = array.shape[0]
    tril_minor_diag_inds = tril_indices_minor_diag(n_steps, 1)
    array[tril_minor_diag_inds] = 0
    return array


def plot_line_graph(
    weight_strat_name,
    otp_data,
    pta_data,
    rounding_epsilon,
    appropriateness_epsilon,
    command,
    n_steps,
    n_rules,
    n_classes,
    class_freq_dist,
    exact=True,
):
    plt.rcParams["text.usetex"] = True
    otp_data = np.array(otp_data)
    pta_data = np.array(pta_data)

    otp_color = "chartreuse"
    pta_color = "cornflowerblue"

    # create filename
    prob_type = "density" if command == "integrate" else "mass"
    bound_or_not = "exact" if exact else "estimate"

    rnd_eps_str = (
        "rndeps_" + str(rounding_epsilon) + "_" if rounding_epsilon > 0 else ""
    )
    app_eps_str = (
        "appeps_" + str(appropriateness_epsilon) + "_"
        if appropriateness_epsilon > 0
        else ""
    )

    fname = "_".join(
        [
            weight_strat_name,
            prob_type,
            bound_or_not,
            "p",
            str(n_rules),
            "k",
            str(n_classes),
            "nsteps",
            str(n_steps),
            "classfreq",
            str(class_freq_dist),
            rnd_eps_str + app_eps_str + "line_graph_plot",
        ]
    )
    fname = fname.replace(".", "_") + ".pdf"
    fname = join("./results/", fname)

    # compute the means of accuracies
    accs_choices = np.linspace(0, 1, n_steps + 2)[1:-1]
    accs_iter = itertools.combinations_with_replacement(accs_choices, n_rules)
    accs_mean = np.array([np.mean(accs) for accs in accs_iter])

    sort_inds = np.argsort(accs_mean)

    # plot
    fig, ax = plt.subplots()
    ax_pta = ax.twinx()
    ln_otp = ax.plot(
        accs_mean[sort_inds],
        otp_data[sort_inds],
        color=otp_color,
        linestyle=LINESTYLE_DICT["dashed"],
        label="Opt./Poss.",
    )
    ln_pta = ax_pta.plot(
        accs_mean[sort_inds],
        pta_data[sort_inds],
        color=pta_color,
        linestyle="dotted",
        label="Poss./All",
    )

    # axes names
    ax.set_xlabel("Mean Accuracy")
    ax_pta.set_ylabel("Poss./All " + prob_type.title())
    ax.set_ylabel("Opt./Poss. " + prob_type.title())

    # legend stuff
    lns = ln_otp + ln_pta
    labs = [str(ln.get_label()) for ln in lns]
    ax.legend(lns, labs, loc="upper left")

    print("saving figure to ", fname)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# for creating a heatmap when n_rules=2
def plot_2d_heatmap(
    weight_strat_name,
    data,
    rounding_epsilon,
    appropriateness_epsilon,
    command,
    n_steps,
    n_rules,
    n_classes,
    class_freq_dist,
    opt_to_poss=True,
    exact=True,
):
    plt.rcParams["text.usetex"] = True
    # create filename
    prob_type = "density" if command == "integrate" else "mass"
    bound_or_not = "exact" if exact else "estimate"
    ratio_type = "opt2pos" if opt_to_poss else "pos2all"

    rnd_eps_str = (
        "rndeps_" + str(rounding_epsilon) + "_" if rounding_epsilon > 0 else ""
    )
    app_eps_str = (
        "appeps_" + str(appropriateness_epsilon) + "_"
        if appropriateness_epsilon > 0
        else ""
    )

    fname = "_".join(
        [
            weight_strat_name,
            prob_type,
            bound_or_not,
            ratio_type,
            "p",
            str(n_rules),
            "k",
            str(n_classes),
            "nsteps",
            str(n_steps),
            "classfreq",
            str(class_freq_dist),
            rnd_eps_str + app_eps_str + "2d_heatmap_plot",
        ]
    )
    fname = fname.replace(".", "_") + ".pdf"
    fname = join("./results/", fname)

    # meshgrid info
    graph_coords = np.linspace(0, 1, n_steps + 1)
    xx, yy = np.meshgrid(graph_coords, graph_coords)

    # colorbar info
    # in some cases, a degenerate combination of rule accs and
    # class frequencies is present (all equal 1/k).
    # in that case, make the value of 1 an outlier.
    data_contains_one = np.any(np.isclose(data, 1))
    if data_contains_one:
        data_without_one = data.copy()
        data_without_one[data == 1] = 0
        data_max_wo_one = np.max(data_without_one)
        vmax = data_max_wo_one
    else:
        vmax = None
    ratio_type = "Optimal/Possible" if opt_to_poss else "Possible/All"
    exact_or_not = "(Exact)" if exact else "(Upper Bound)"
    prob_type = "Mass" if command == "count" else " Density"
    cbarlabel = " ".join([ratio_type, prob_type, exact_or_not])

    # plot
    cmap = plt.get_cmap("viridis")
    cmap.set_over("red")
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(xx, yy, data, cmap=cmap, vmax=vmax)
    cbar = fig.colorbar(mesh, ax=ax, extend="max")

    # axes names
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xlabel(r"$b_1^*$")
    ax.set_ylabel(r"$b_2^*$")

    print("saving figure to ", fname)
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # replot = True
    replot = False
    n_rules = 2
    n_classes = 2
    use_specialists = False
    n_steps = 9
    n_datapoints = 400
    weights_used = "ocds"
    delete_hrep_files = True
    delete_latte_outputs = True
    appropriateness_epsilon = gmpy2.mpq(0, 1)
    # appropriateness_epsilon = gmpy2.mpq(1, 10)

    # count takes a really long time even for something simple
    # like n_rules = 3, n_classes = 2, when counting
    # the number of patterns in T^0, the set of possible patterns.
    # this is because T^0 has an extremely large number of vertices

    commands = ["integrate", "count"]
    # commands = ["integrate"]
    # commands = ["count"]

    # generate the pairs of accuracies, drop 0 and 1 b/c the counter can't
    # accommodate those accuracies
    # accs_choices = np.linspace(0, 1, n_steps + 2)[1:-1]
    accs_denom = gmpy2.mpq(n_steps + 1)
    accs_nums = np.arange(1, n_steps + 1).astype(int)
    accs_choices = [accs_num / accs_denom for accs_num in accs_nums]
    # this has a lot of repeats because the order of the rules doesn't matter
    # accs_list = list(itertools.product(accs_choices, repeat=n_rules))
    accs_list = list(itertools.combinations_with_replacement(accs_choices, n_rules))
    # generate class frequencies.  Due to symmetry, we only go up to 1/n_classes
    # class_freq_probs = np.linspace(0, 1 / n_classes, n_steps + 1)[1:]
    class_freqs_denom = gmpy2.mpq(1, n_classes * n_steps)
    class_freqs_probs = [
        class_freqs_num / class_freqs_denom for class_freqs_num in accs_nums
    ]
    class_freqs = [np.array([mass, 1 - mass]) for mass in class_freqs_probs]
    print("Overriding class_freqs and using the uniform distribution.")
    class_freqs = [np.array([gmpy2.mpq(1, n_classes) for _ in range(n_classes)])]

    placeholder_accs = np.array([gmpy2.mpq(1, n_classes) for _ in range(n_rules)])
    placeholder_class_freqs = np.array(
        [gmpy2.mpq(1, n_classes) for _ in range(n_classes)]
    )

    # placeholder_accs = [gmpy2.mpq(1, n_classes)] * n_rules
    # placeholder_class_freqs = [gmpy2.mpq(1, n_classes)] * n_classes

    ocds_weights = OneCoinDawidSkeneWeights(
        placeholder_accs, placeholder_class_freqs, n_classes
    )
    mv_weights = MajorityVoteWeights(placeholder_accs, placeholder_class_freqs)

    if weights_used == "ocds":
        weights = ocds_weights
    elif weights_used == "mv":
        weights = mv_weights
    else:
        raise NotImplementedError("Weights ", weights_used, " not implemented!")
    weight_strat_name = weights.get_name_shorthand()

    compute_exact_cond_dist = weights.exp_weights_are_rational()
    # compute_exact_cond_dist = False

    pattern_counter = WeightOptimality(
        weights,
        n_rules,
        n_classes,
        placeholder_accs,
        placeholder_class_freqs,
        # enums_to_skip=[Polytope.POSSIBLE_PATTERNS_ALL],
        enums_to_skip=[Polytope.POSSIBLE_PATTERNS_VREP_REPARAM],
        use_specialists=use_specialists,
        compute_exact_cond_dist=compute_exact_cond_dist,
        delete_outputs=delete_latte_outputs,
        appropriateness_epsilon=appropriateness_epsilon,
        datapoints=n_datapoints,
    )

    polytopes: List[Polytope] = []
    for command in commands:
        for class_freq_dist in class_freqs:
            if not replot:
                for accs in accs_list:
                    accs = np.array(accs)
                    pattern_counter.update_accs_class_freqs(accs, class_freq_dist)
                    file_enum, polytopes = pattern_counter.construct_inputs()
                    pattern_counter.count_or_integrate_patterns(
                        command, file_enum, delete_hrep_file=delete_hrep_files
                    )

            (
                otp_data,
                atp_data,
                rounding_epsilon,
                appropriateness_epsilon,
                polytope_avg_dims,
            ) = read_data(
                pattern_counter,
                class_freq_dist,
                n_rules,
                n_steps,
                command,
                polytopes,
                reshape_results_to_matrix=n_rules == 2,
            )
            if n_rules == 2:
                plot_2d_heatmap(
                    weight_strat_name,
                    otp_data,
                    rounding_epsilon,
                    appropriateness_epsilon,
                    command,
                    n_steps,
                    n_rules,
                    n_classes,
                    class_freq_dist,
                    exact=compute_exact_cond_dist,
                )
                plot_2d_heatmap(
                    weight_strat_name,
                    atp_data,
                    rounding_epsilon,
                    appropriateness_epsilon,
                    command,
                    n_steps,
                    n_rules,
                    n_classes,
                    class_freq_dist,
                    opt_to_poss=False,
                    exact=compute_exact_cond_dist,
                )
            else:
                plot_line_graph(
                    weight_strat_name,
                    otp_data,
                    atp_data,
                    rounding_epsilon,
                    appropriateness_epsilon,
                    command,
                    n_steps,
                    n_rules,
                    n_classes,
                    class_freq_dist,
                    exact=compute_exact_cond_dist,
                )
