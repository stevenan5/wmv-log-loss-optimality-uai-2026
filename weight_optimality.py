import os
import copy
import subprocess
from enum import Enum
from os.path import join
from typing import List

import numpy as np
import gmpy2
from gmpy2 import mpq, mpfr
from scipy.io import loadmat, savemat

import cdd.gmp as cdd
from cdd import RepType

from bf import BFJoint
from util import _check_shape, stars_and_bars, scaled_simplex_vol
from weighting_strategy import WeightingStrategy


# These enums characterize the counting procedures that are run
# when counting patterns.  Here, we assume that the LF/rule accs
# are fixed as well as the class frequencies.  Moreover, we
# assume the number of datapoints is fixed.
class Polytope(Enum):
    # Fix weights that only depend on the rule accs/class freqs
    # this counts the number of patterns where those weights are optimal
    EPS_OPT_PATTERNS = 0
    # scaled probability simplex, mostly as a sanity check.
    # we'll use the result from scaled_simplex_vol in general
    POSSIBLE_PATTERNS_ALL = 1
    # Count the number of patterns consistent with the LF accs
    # and class frequencies by converting polytope from h-rep to v-rep
    # and then applying a transformation to the vertices, then
    # converting back to h-rep.
    POSSIBLE_PATTERNS_VREP = 2
    # do the same as above, but rewrite the polytope constraint matrix
    # so the pattern masses are variables.  Empirical exps show that
    # this results in less intermediate vertices.
    POSSIBLE_PATTERNS_VREP_REPARAM = 3


POLYTOPE_FILE_NAMES = {
    Polytope.EPS_OPT_PATTERNS: "eps_opt",
    Polytope.POSSIBLE_PATTERNS_ALL: "all_possible",
    Polytope.POSSIBLE_PATTERNS_VREP: "possible_vrep",
    Polytope.POSSIBLE_PATTERNS_VREP_REPARAM: "possible_vrep_reparam",
}

POLYTOPE_INPUT_TYPES = {
    # can also be "latte", need to uncomment lines in
    # `construct_input_helper`if so
    Polytope.EPS_OPT_PATTERNS: "cdd",
    Polytope.POSSIBLE_PATTERNS_ALL: "cdd",
    Polytope.POSSIBLE_PATTERNS_VREP: "cdd",
    Polytope.POSSIBLE_PATTERNS_VREP_REPARAM: "cdd",
}

DEFAULT_EXTRA_ARGS = [
    "--triangulation=4ti2",
    "--dualization=4ti2",
    "--compute-vertex-cones=4ti2",
]

POLYTOPE_COUNT_EXTRA_ARGS = {
    Polytope.EPS_OPT_PATTERNS: [],
    Polytope.POSSIBLE_PATTERNS_ALL: [],
    Polytope.POSSIBLE_PATTERNS_VREP: [
        "--irrational-all-primal",
        "--maxdet=1000",
        "--exponential",
    ],
    Polytope.POSSIBLE_PATTERNS_VREP_REPARAM: [
        "--irrational-all-primal",
        "--maxdet=1000",
        "--exponential",
    ],
}


class WeightOptimality(BFJoint):
    def __init__(
        self,
        weighting_strategy: WeightingStrategy,
        n_rules,
        n_classes,
        true_accs,
        true_class_freqs,
        datapoints=10**3,
        decimal_places=4,
        rounding_epsilon: float = -1,
        appropriateness_epsilon: gmpy2.mpq = mpq(0, 1),
        enums_to_skip: List[Polytope] = [
            Polytope.POSSIBLE_PATTERNS_ALL,
            Polytope.POSSIBLE_PATTERNS_VREP_REPARAM,
        ],
        compute_exact_cond_dist=True,
        use_specialists=False,
        delete_outputs=False,
        dir="./results",
    ):
        super().__init__(
            n_rules,
            n_classes,
            true_accs,
            true_class_freqs,
            use_specialists=use_specialists,
        )
        self.weighting_strategy = weighting_strategy
        # enforce a minimum of two decimal places so the epsilon will not exceed 0.1
        # by default
        if decimal_places <= 1 and rounding_epsilon == -1:
            raise ValueError(
                "Number of decimal places cannot be less than 2 when epsilon is unset."
            )
        self.decimal_places = decimal_places
        # if rounding_epsilon is not set, then we will default to the smallest
        # possible rounding_epsilon that captures the inaccuracy of our
        # rationalization.  The maximum rounding error is
        # 10**(decimal_places - 1)
        self.appropriateness_epsilon = (
            # going straight from mpq to float can sometimes lead to stuff like
            # 0.1 represented as 0.09999999999
            float(gmpy2.mpfr(appropriateness_epsilon))
            if not compute_exact_cond_dist
            else appropriateness_epsilon
        )

        self.compute_exact_cond_dist = compute_exact_cond_dist
        if self.compute_exact_cond_dist:
            if self.weighting_strategy.exp_weights_are_rational():
                self.rounding_epsilon = 0
                print(
                    "Overriding self.rounding_epsilon to be 0 because weighting strategy "
                    + "supports exact computation of conditional distribution."
                    + "self.decimal_places will also be ignored."
                )
            else:
                raise ValueError(
                    "Cannot compute exact conditional distribution with weighting strategy ",
                    self.weighting_strategy.get_name_shorthand,
                )
        else:
            # cast the accuracies and class frequencies to floats
            self.true_accs = self.true_accs.astype(float)
            self.true_class_freqs = self.true_class_freqs.astype(float)

            self.rounding_epsilon = (
                10 ** (1 - self.decimal_places)
                if rounding_epsilon == -1
                else rounding_epsilon
            )

        self.polytope_dims = {poly_enum: -1 for poly_enum in Polytope}

        self.marginal_equality = (
            self.rounding_epsilon == 0 and self.appropriateness_epsilon == 0
        )
        self.total_epsilon = self.rounding_epsilon + self.appropriateness_epsilon

        self.datapoints = datapoints
        self.delete_outputs = delete_outputs
        self.enums_to_skip = enums_to_skip
        self.update_accs_class_freqs(true_accs, true_class_freqs)

        if (
            len(
                set(self.enums_to_skip)
                - set(
                    [
                        Polytope.POSSIBLE_PATTERNS_ALL,
                        Polytope.POSSIBLE_PATTERNS_VREP,
                        Polytope.POSSIBLE_PATTERNS_VREP_REPARAM,
                    ]
                )
            )
            > 0
        ):
            raise ValueError(
                "Can only skip the following enums",
                Polytope.POSSIBLE_PATTERNS_ALL,
                Polytope.POSSIBLE_PATTERNS_VREP,
                Polytope.POSSIBLE_PATTERNS_VREP_REPARAM,
            )

        self.dir = join(dir, self.bookkeeping_fname_prefix())
        WeightOptimality.check_inputs(self)
        self.bookkeeping_exists = False
        self.bookkeeping_init()

        # stuff for convenience
        self.pattern_constraints_dense = self.pattern_constraints.toarray().astype(int)

    def check_inputs(self):
        _check_shape(self.rule_weights, (self.n_rules,))
        _check_shape(self.class_freqs_weights, (self.n_classes,))

    def refresh_weights(self):
        self.rule_weights = self.weighting_strategy.construct_rule_weights()
        self.class_freqs_weights = (
            self.weighting_strategy.construct_class_freqs_weights()
        )

    def update_accs_class_freqs(self, accs, class_freqs):
        self.true_accs = accs
        self.true_class_freqs = class_freqs
        self.weighting_strategy.update_accs_class_freqs(accs, class_freqs)
        # if compute_exact_cond_dist is true, then we need to do more checks
        if self.compute_exact_cond_dist:
            # if the provided accs and class freqs are not mpq, then also fail
            all_accs_are_mpq = np.all([isinstance(acc, mpq) for acc in self.true_accs])
            all_class_freqs_are_mpq = np.all(
                [isinstance(cf, mpq) for cf in self.true_class_freqs]
            )
            if not (all_accs_are_mpq and all_class_freqs_are_mpq):
                raise ValueError(
                    "Must provide accs and class_freqs as gmpy2.mpq"
                    + " objects when compute_exact_cond_dist is True."
                )

            # make sure that the number of datapoint is a multiple of
            # all denominators of accuracies and class frequencies
            all_denoms = [acc.denominator for acc in self.true_accs] + [
                cf.denominator for cf in self.true_class_freqs
            ]
            prod = np.prod(all_denoms) * (self.n_classes - 1) ** self.n_rules
            if self.datapoints % prod != 0:
                raise ValueError(
                    "Number of datapoints ",
                    self.datapoints,
                    "must be a multiple of the product",
                    " of all accs/class freqs denominators times "
                    f"(n_classes - 1) ** n_rules, i.e. {prod}",
                    " when compute_exact_cond_dist is True.",
                )
        else:
            self.true_accs = self.true_accs.astype(float)
            self.true_class_freqs = self.true_class_freqs.astype(float)

        self.refresh_weights()

    def construct_conditional_dist(self):
        if self.compute_exact_cond_dist:
            # make this a column vector
            rule_class_freqs_exp_weights = self.weighting_strategy.get_exp_weights()[
                :, None
            ]
            dense_rule_pred_const = self.rule_pred_constraints.toarray()
            dense_class_freqs_const = self.class_freq_constraints.toarray()
            # cast to int to prevent mpfr from appearing -- we want to keep mpq's
            pred_and_class_const = np.vstack(
                (dense_rule_pred_const, dense_class_freqs_const)
            ).astype(int)
            inv_const = 1 - pred_and_class_const
            weighted_const = rule_class_freqs_exp_weights * pred_and_class_const
            # add 1's to where the 0s are because we want to take a product
            weighted_const += inv_const
            numerators = np.prod(weighted_const, axis=0)
            reshaped_nums = numerators.reshape((-1, self.n_classes))
            denominators = np.repeat(reshaped_nums.sum(axis=1), self.n_classes)
            self.cond_dist = numerators / denominators

        else:
            cond_scores = self.compute_conditional_scores(
                self.rule_weights, self.class_freqs_weights
            )
            self.cond_dist = self.conditional_softmax(cond_scores)

    def construct_conditional_constraint_matrix(self):
        res = np.zeros((self.n_rules + self.n_classes, self.n_patterns))

        rule_pred_const = self.rule_pred_constraints.toarray()
        patt_const = self.pattern_constraints_dense
        if self.compute_exact_cond_dist:
            res = res.astype(mpq)
            rule_pred_const = rule_pred_const.astype(int)
            patt_const = patt_const.astype(int)

        cond_dist_w_rule_pred_const = rule_pred_const * self.cond_dist
        res[: self.n_rules, :] = cond_dist_w_rule_pred_const @ patt_const.T

        # make rows for the class frequencies
        reshaped_cond_dist = self.cond_dist.reshape((self.n_patterns, self.n_classes))
        for ell in range(self.n_classes):
            res[self.n_rules + ell, :] = reshaped_cond_dist[:, ell]

        self.cond_constraint_matrix = res

    def rationalize_values_DEPRECATED(self):
        scale = 10**self.decimal_places
        rounded_constraints = np.round(scale * self.cond_constraint_matrix).astype(int)
        rhs = np.concatenate((self.true_accs, self.true_class_freqs))
        # find the GCD of every row to try and get an easy simplification
        # unclear if smaller numbers will make the lattice algorithms faster
        if self.marginal_equality:
            self.rationalized_constraints = rounded_constraints
            self.rationalized_rhs = np.round(scale * self.datapoints * rhs).astype(int)
        else:
            self.rationalized_constraints_ub = rounded_constraints
            self.rationalized_constraints_lb = rounded_constraints
            self.rationalized_rhs_ub = np.round(
                scale * self.datapoints * (rhs + self.total_epsilon)
            ).astype(int)
            self.rationalized_rhs_lb = np.round(
                scale * self.datapoints * (rhs - self.total_epsilon)
            ).astype(int)

        for j in range(self.n_rules + self.n_classes):
            if self.marginal_equality:
                row_and_rhs = np.concatenate(
                    (self.rationalized_constraints[j, :], [self.rationalized_rhs[j]])
                )
                gcd = np.gcd.reduce(row_and_rhs)
                self.rationalized_constraints[j, :] //= gcd
                self.rationalized_rhs[j] //= gcd
            else:
                row_and_rhs_ub = np.concatenate(
                    (
                        self.rationalized_constraints_ub[j, :],
                        [self.rationalized_rhs_ub[j]],
                    )
                )
                gcd_ub = np.gcd.reduce(row_and_rhs_ub)
                self.rationalized_constraints_ub[j, :] //= gcd_ub
                self.rationalized_rhs_ub[j] //= gcd_ub

                row_and_rhs_lb = np.concatenate(
                    (
                        self.rationalized_constraints_lb[j, :],
                        [self.rationalized_rhs_lb[j]],
                    )
                )
                gcd_lb = np.gcd.reduce(row_and_rhs_lb)
                self.rationalized_constraints_lb[j, :] //= gcd_lb
                self.rationalized_rhs_lb[j] //= gcd_lb

    def rationalize_values(self):
        if self.compute_exact_cond_dist:
            # this is just for safety, we'll actually just skip the scaling step below
            scale = 1
        else:
            scale = mpq(10**self.decimal_places)
        lhs = self.cond_constraint_matrix
        rhs = np.vstack((self.true_accs[:, None], self.true_class_freqs[:, None]))
        if self.marginal_equality:
            rhs *= self.datapoints
            matrix = np.hstack([rhs, -1 * lhs])
        else:
            rhs_both = self.datapoints * np.vstack(
                [rhs + self.total_epsilon, self.total_epsilon - rhs]
            )
            lhs_both = np.vstack([-1 * lhs, lhs])
            matrix = np.hstack([rhs_both, lhs_both])

        # convert the matrix into mpq and then multiply by scale and then truncate
        mat_list = matrix.tolist()
        mat_list_mpq = [[mpq(ele) for ele in line] for line in mat_list]
        if self.compute_exact_cond_dist:
            mat_list_mpq_trunc = mat_list_mpq
        else:
            mat_list_mpq_scaled = [
                [gmpy2.mul(ele, scale) for ele in line] for line in mat_list_mpq
            ]
            mat_list_mpq_trunc = [
                [mpq(gmpy2.trunc(mpfr(ele))) for ele in line]
                for line in mat_list_mpq_scaled
            ]

        mat_list_rationalized = [
            self.convert_rational_to_int(line) for line in mat_list_mpq_trunc
        ]
        return mat_list_rationalized

    def construct_hrep_input_fname(self, file_enum: int, polytope: Polytope):
        if POLYTOPE_INPUT_TYPES[polytope] == "latte":
            file_ext = ".hrep.latte"
        elif POLYTOPE_INPUT_TYPES[polytope] == "cdd":
            file_ext = ".ine"
        else:
            file_ext = ".txt"

        return join(
            self.dir,
            "input_" + str(file_enum) + "_run_" + str(polytope.value) + file_ext,
        )

    def construct_fix_cond_pattern_counting_matrix_latte(self):
        # the extra 1 for constraints is so the variables are a distribution.
        n_constraints = self.n_rules + self.n_classes + 1
        # must be negative since for Ax <= b, we need to input as b -A
        pattern_dist_row = -1 * np.ones((1, self.n_patterns))
        if self.marginal_equality:
            # LattE indices start with 1
            indices_for_equality = np.arange(n_constraints) + 1
            rhs = np.concatenate([self.rationalized_rhs, [self.datapoints]])
            lhs = np.vstack((-1 * self.rationalized_constraints, pattern_dist_row))
        else:
            n_constraints += self.n_rules + self.n_classes
            indices_for_equality = np.array([n_constraints])
            rhs = np.concatenate(
                [
                    self.rationalized_rhs_ub,
                    -1 * self.rationalized_rhs_lb,
                    [self.datapoints],
                ]
            )
            lhs = np.vstack(
                (
                    -1 * self.rationalized_constraints_ub,
                    self.rationalized_constraints_lb,
                    pattern_dist_row,
                )
            )
        array_to_write = np.hstack((rhs[:, None], lhs))

        indices_for_nonneg = np.arange(self.n_patterns) + 1
        n_dims = self.n_patterns

        return (
            array_to_write,
            n_constraints,
            indices_for_equality,
            indices_for_nonneg,
            n_dims,
        )

    def construct_input_helper(self, file_enum: int, polytope: Polytope):
        # reset the dictioanry every time we make new inputs
        self.polytope_dims = {poly_enum: -1 for poly_enum in Polytope}
        if polytope == Polytope.EPS_OPT_PATTERNS:
            self.construct_conditional_dist()
            self.construct_conditional_constraint_matrix()
            cdd_matrix_partial = self.rationalize_values()
            raw_cdd_output, polytope_dim = (
                self.construct_cond_pattern_counting_matrix_cdd(cdd_matrix_partial)
            )
            self.write_cdd_output_file(
                raw_cdd_output, polytope_dim, file_enum, polytope
            )
            # To use this you will need to change
            # POLYTOPE_INPUT_TYPES[Polytope.EPS_OPT_PATTERNS] to equal "latte"
            # self.rationalize_values_DEPRECATED()
            # (
            #     array_to_write,
            #     n_constraints,
            #     indices_for_equality,
            #     indices_for_nonneg,
            #     n_dims,
            # ) = self.construct_fix_cond_pattern_counting_matrix_latte()
            #
            # self.construct_latte_hrep_file(
            #     array_to_write,
            #     n_constraints,
            #     indices_for_equality,
            #     indices_for_nonneg,
            #     n_dims,
            #     file_enum,
            #     polytope,
            # )
        elif polytope == Polytope.POSSIBLE_PATTERNS_ALL:
            raw_cdd_output = self.construct_scaled_probability_simplex()
            polytope_dim = self.n_patterns - 1
            self.write_cdd_output_file(
                raw_cdd_output, polytope_dim, file_enum, polytope
            )
        elif polytope in [
            Polytope.POSSIBLE_PATTERNS_VREP,
            Polytope.POSSIBLE_PATTERNS_VREP_REPARAM,
        ]:
            raw_cdd_output, polytope_dim = self.construct_joint_pattern_counting_matrix(
                file_enum, polytope
            )
            self.write_cdd_output_file(
                raw_cdd_output, polytope_dim, file_enum, polytope
            )
        else:
            raise NotImplementedError(
                "No input construction for polytope ", polytope.name, " exists"
            )

        self.polytope_dims[polytope] = polytope_dim

    def construct_latte_hrep_file(
        self,
        array_to_write,
        n_constraints,
        indices_for_equality,
        indices_for_nonneg,
        n_dims,
        file_enum,
        polytope,
    ):
        fname = self.construct_hrep_input_fname(file_enum, polytope)

        equality_constraint = (
            " ".join(
                ["linearity", str(len(indices_for_equality))]
                + list(indices_for_equality.astype(str))
            )
            + "\n"
        )
        if indices_for_nonneg is not None:
            nonneg_constraint = " ".join(
                ["nonnegative", str(len(indices_for_nonneg))]
                + list(indices_for_nonneg.astype(str))
            )
        else:
            nonneg_constraint = ""

        footer = equality_constraint + nonneg_constraint

        # the LattE input file requirements wants numer of constraints and
        # number of dimensions + 1
        # the footer will contain a requirement that the variables be
        # non-negative
        header = " ".join([str(n_constraints), str(n_dims + 1)])

        np.savetxt(
            fname,
            array_to_write,
            header=header,
            footer=footer,
            comments="",
            fmt=f"%{self.decimal_places + int(np.floor(np.log10(self.datapoints)))}.0f",
        )

    def write_cdd_output_file(
        self, cdd_output: str, polytope_dim: int, file_enum: int, polytope: Polytope
    ):
        fname = self.construct_hrep_input_fname(file_enum, polytope)
        # change real to rational to LattE doesn't complain
        if "real" in cdd_output:
            cdd_output = cdd_output.replace("real", "rational")

        # generate preamble
        if polytope == Polytope.EPS_OPT_PATTERNS:
            weighting_strat_string = f"\n* weighting strategy: {self.weighting_strategy.get_name_shorthand()}"
        else:
            weighting_strat_string = ""
        preamble = "* Polytope enum: " + polytope.name
        preamble += "\n* ".join(
            [
                "",
                "Construction Information:",
                f"n_classes: {self.n_classes}",
                f"n_rules: {self.n_rules}",
                f"n_patterns (dimensions): {self.n_patterns}",
                f"polytope dimension: {polytope_dim}",
                f"true accs: {self.true_accs}",
                f"true class_freqs: {self.true_class_freqs}",
                f"rounding epsilon: {self.rounding_epsilon}",
                f"appropriateness epsilon: {self.appropriateness_epsilon}",
            ]
        )
        output = preamble + weighting_strat_string + "\n" + cdd_output
        with open(fname, "w") as f:
            f.write(output)

    def check_cdd_output_is_integral(self, cdd_out: str) -> bool:
        return "/" not in cdd_out

    def convert_cdd_output_to_ints(self, cdd_output: str) -> str:
        out_lines = []
        lines = str.splitlines(cdd_output)

        begin_ind = -1
        begin_passed = False

        for i, line in enumerate(lines):
            # don't touch the start of file until we get to the
            # actual constraints
            if "begin" in line:
                begin_passed = True
                out_lines.append(line)
                begin_ind = i
                continue
            if begin_passed and i == begin_ind + 1:
                out_lines.append(line)
                continue

            if i < len(lines) - 1:
                if "/" not in line:
                    out_lines.append(line)
                    continue
                else:
                    numbers_str = line.split(" ")
                    # cdd starts line with an empty space, which leads the first element to
                    # be the empty string.  we'll add this back at the end
                    numbers_str = numbers_str[1:]
                    numbers_gmp = [mpq(num) for num in numbers_str]
                    numbers_gmp = self.convert_rational_to_int(numbers_gmp)
                    numbers_row = " ".join([""] + [str(num) for num in numbers_gmp])
                    out_lines.append(numbers_row)
            else:
                # append the `end` keyword to finish
                out_lines.append(line)

        return "\n".join(out_lines)

    def convert_rational_to_int(self, inputs: List[mpq]):
        numbers_den = [num.denominator for num in inputs]
        lcm = gmpy2.lcm(*numbers_den)
        return [gmpy2.mul(num, lcm) for num in inputs]

    def construct_joint_orig_pattern_counting_cdd_matrix(self):
        # we require that
        # self.datapoints * {self.true_accs, self.class_freqs}
        # is an integer.  This is so we have rational constraints
        # and can use equality constraints for cdd/latte

        # we will have 4 sets of constraints
        # 1. rule predictions (accuracies)
        # 2. class frequencies
        # 3. distribution (elements sum to self.datapoints * 1)
        # 4. non-negativity for each variable
        def scaling_helper(x):
            return mpq(self.datapoints) * x

        n_joint_vars = self.n_patterns * self.n_classes
        if self.appropriateness_epsilon == 0:
            rhs = np.vstack(
                [
                    self.true_accs[:, None],
                    self.true_class_freqs[:, None],
                    [[1]],
                    np.zeros((n_joint_vars, 1)).astype(int),
                ]
            )
            if not self.compute_exact_cond_dist:
                rhs = self.datapoints * np.round(
                    rhs, decimals=self.decimal_places
                ).astype(int)
            else:
                rhs = np.array(np.vectorize(scaling_helper)(rhs))
            # the expected format for cdd is [b A] for b <= Ax
            # and b=Ax, hence we need to negate the RHS.
            matrix = np.vstack(
                [
                    -1 * self.rule_pred_constraints.toarray().astype(int),
                    -1 * self.class_freq_constraints.toarray().astype(int),
                    -1 * np.ones((1, n_joint_vars)),
                    np.eye(n_joint_vars),
                ]
            ).astype(int)

            n_equality_inds = self.n_rules + self.n_classes + 1
            indices_for_equality = np.arange(n_equality_inds)
        else:
            if self.compute_exact_cond_dist:
                accs_ub = self.true_accs + self.appropriateness_epsilon
                accs_ub = np.array(
                    [[acc_ub if acc_ub < 1 else mpq(1, 1)] for acc_ub in accs_ub]
                )

                accs_lb = self.true_accs - self.appropriateness_epsilon
                accs_lb = np.array(
                    [[acc_lb if acc_lb > 0 else mpq(0, 1)] for acc_lb in accs_lb]
                )
                class_freqs_ub = self.true_class_freqs + self.appropriateness_epsilon
                class_freqs_ub = np.array(
                    [[cf_ub if cf_ub < 1 else mpq(1, 1)] for cf_ub in class_freqs_ub]
                )
                class_freqs_lb = self.true_class_freqs - self.appropriateness_epsilon
                class_freqs_lb = np.array(
                    [[cf_lb if cf_lb > 0 else mpq(0, 1)] for cf_lb in class_freqs_lb]
                )
            else:
                accs_ub = np.clip(
                    self.true_accs[:, None] + self.appropriateness_epsilon, 0, 1
                )
                accs_lb = np.clip(
                    self.true_accs[:, None] - self.appropriateness_epsilon, 0, 1
                )
                class_freqs_ub = np.clip(
                    self.true_class_freqs[:, None] + self.appropriateness_epsilon, 0, 1
                )
                class_freqs_lb = np.clip(
                    self.true_class_freqs[:, None] - self.appropriateness_epsilon, 0, 1
                )
            rhs = np.vstack(
                [
                    accs_ub,
                    -1 * accs_lb,
                    class_freqs_ub,
                    -1 * class_freqs_lb,
                    [[1]],
                    np.zeros((n_joint_vars, 1)).astype(int),
                ]
            )
            if not self.compute_exact_cond_dist:
                rhs = (
                    self.datapoints
                    * np.round(
                        rhs,
                        decimals=self.decimal_places,
                    )
                ).astype(int)
            else:
                rhs = np.vectorize(scaling_helper)(rhs)
            # the expected format for cdd is [b A] for b <= Ax
            # and b=Ax, hence we need to negate the RHS.
            matrix = np.vstack(
                [
                    -1 * self.rule_pred_constraints.toarray(),
                    self.rule_pred_constraints.toarray(),
                    -1 * self.class_freq_constraints.toarray(),
                    self.class_freq_constraints.toarray(),
                    -1 * np.ones((1, n_joint_vars)),
                    np.eye(n_joint_vars),
                ]
            ).astype(int)

            indices_for_equality = [2 * (self.n_rules + self.n_classes)]

        return rhs, matrix, indices_for_equality

    def construct_joint_reparam_pattern_counting_cdd_matrix(self):
        if self.appropriateness_epsilon != 0:
            raise NotImplementedError(
                "Reparameterized pattern counting matrix does not "
                + "support appropriateness epsilon >0.  "
                + "Please use original paramterization polytope"
            )
        # here, we write the constraints in a different way so that the
        # pattern variables are explicitly instantiated.
        # the variables will be in n_classes + 1 blocks (n_patterns many blocks)
        # they will be (alpha_t, beta_t1, beta_t2, beta_t3,...)
        # where alpha_t is the pattern mass and alpha_t - beta_t1 is Pr(y=1 | pattern t)

        # like before, we'll have the accuracies and class freqs
        # then, the rhs entries are as follows.  The matrices will correspond with this
        # 1. 1, the sum of *pattern* masses
        # 2. 0, ensure that the sum of  beta_t,ell over ell in {1, ..., k}
        #       is equal to (n_classes - 1) * alpha_t
        # 3. 0 ensure that none of the beta variables exceed their respective alpha vals
        # 4. 0 to ensure that all variables are non-negative
        n_joint_vars = self.n_patterns * (self.n_classes + 1)
        rhs = (
            self.datapoints
            * np.round(
                np.vstack(
                    [
                        self.true_accs[:, None],
                        self.true_class_freqs[:, None],
                        [[1]],  # bullet 1
                        np.zeros((self.n_patterns, 1)),
                        np.zeros((self.n_patterns * self.n_classes, 1)),
                        np.zeros((n_joint_vars, 1)),
                    ]
                ),
                decimals=self.decimal_places,
            )
        ).astype(int)

        # define a helper function to interleave columns
        # for a block of n_classes elements, (a_1, ..., a_k) (k=n_classes)
        # we append a column on the left to get (a_0, a_1, ..., a_k)

        def interleave(input, scale=1, mode="zeros"):
            # we expect input to have columns that's a multiple of n_classes
            # mode determines what is interleaved
            # either zeros, or the negated row sum
            n_rows = input.shape[0]
            reshaped_input = np.reshape(input, (n_rows, self.n_classes, -1), order="F")
            # now compute the array to attach
            attachment_shape = (n_rows, 1, reshaped_input.shape[2])
            if mode == "zeros":
                left_array = np.zeros(attachment_shape)
            elif mode == "ones":
                left_array = np.ones(attachment_shape)
            elif mode == "sum_clip":
                left_array = np.clip(
                    reshaped_input.sum(axis=1, keepdims=True), min=0, max=1
                )
            else:
                raise NotImplementedError(
                    "mode ", mode, "for interleave is not implemented!"
                )

            stacked_array = np.hstack((scale * left_array, reshaped_input))

            return np.reshape(stacked_array, (n_rows, -1), order="F")

        rule_and_class_freq_constraints = np.vstack(
            (
                self.rule_pred_constraints.toarray(),
                self.class_freq_constraints.toarray(),
            )
        )

        mat0 = interleave(
            rule_and_class_freq_constraints,
            mode="sum_clip",
            scale=-1,
        )
        mat1 = interleave(
            np.zeros((1, self.n_classes * self.n_patterns)),
            mode="ones",
            scale=-1,
        )
        mat2 = interleave(
            self.pattern_constraints_dense,
            scale=1 - self.n_classes,
            mode="sum_clip",
        )
        mat3 = -1 * interleave(
            np.eye(self.n_classes * self.n_patterns), scale=-1, mode="sum_clip"
        )

        matrix = np.vstack(
            [
                mat0,
                mat1,  # bullet 1
                mat2,  # bullet 2
                mat3,  # bullet 3
                np.eye(n_joint_vars),  # bullet 4
            ]
        ).astype(int)

        n_equality_inds = self.n_rules + self.n_classes + 1 + self.n_patterns
        inds_for_equality = np.arange(n_equality_inds)

        return rhs, matrix, inds_for_equality

    def construct_cond_pattern_counting_matrix_cdd(
        self, cdd_partial_matrix, return_canonicalize=False
    ):
        n_vars = len(cdd_partial_matrix[0]) - 1
        # construct distribution constraints
        dist_lhs = np.vstack((np.eye(n_vars), np.ones((1, n_vars))))
        dist_rhs = np.vstack((np.zeros((n_vars, 1)), [[-1 * self.datapoints]]))
        dist_matrix = np.hstack((dist_rhs, dist_lhs))
        dist_matrix_list = dist_matrix.tolist()
        dist_matrix_list_mpq = [[mpq(ele) for ele in line] for line in dist_matrix_list]
        cdd_input = cdd_partial_matrix + dist_matrix_list_mpq
        equality_inds = [len(cdd_input) - 1]
        if self.compute_exact_cond_dist and self.appropriateness_epsilon == 0:
            equality_inds += list(np.arange(self.n_rules + self.n_classes))

        cdd_matrix = cdd.matrix_from_array(cdd_input, equality_inds, RepType.INEQUALITY)
        poly = cdd.polyhedron_from_matrix(cdd_matrix)
        hrep = cdd.copy_inequalities(poly)
        # vrep = cdd.copy_generators(poly)
        cdd.matrix_canonicalize(cdd_matrix)
        canon_linset_size = len(cdd_matrix.lin_set)
        # dimension of a polytope is the number of variables minus
        # the dimension of the null space of the equality constraints.
        # by calling canonicalize, all redundancies have been removed
        polytope_dim = len(cdd_input[0]) - 1 - canon_linset_size
        poly_canon = cdd.polyhedron_from_matrix(cdd_matrix)
        hrep_canon = cdd.copy_inequalities(poly_canon)

        if return_canonicalize:
            return str(hrep_canon), polytope_dim
        else:
            return str(hrep), polytope_dim

    def construct_scaled_probability_simplex(self):
        n_vars = self.n_patterns
        rhs = np.vstack(([[-1 * self.datapoints]], np.zeros((n_vars, 1))))
        lhs = np.vstack((np.ones((1, n_vars)), np.eye(n_vars)))
        cdd_matrix_input = list(np.hstack([rhs, lhs]).astype(int))
        inds_for_equality = set([0])
        cdd_matrix = cdd.matrix_from_array(
            cdd_matrix_input, inds_for_equality, RepType.INEQUALITY
        )
        poly = cdd.polyhedron_from_matrix(cdd_matrix)

        pattern_hrep = str(cdd.copy_inequalities(poly))
        return pattern_hrep

    def construct_joint_pattern_counting_matrix(
        self, file_enum: int, polytope: Polytope
    ):
        if polytope == Polytope.POSSIBLE_PATTERNS_VREP:
            construct_cdd_input = self.construct_joint_orig_pattern_counting_cdd_matrix
        elif polytope == Polytope.POSSIBLE_PATTERNS_VREP_REPARAM:
            construct_cdd_input = (
                self.construct_joint_reparam_pattern_counting_cdd_matrix
            )
        else:
            raise NotImplementedError(
                "Vertex transformation for polytope ",
                polytope.name,
                " is not implemented!",
            )
        rhs, matrix, inds_for_equality = construct_cdd_input()
        cdd_matrix_input = list(np.hstack([rhs, matrix]))
        cdd_matrix = cdd.matrix_from_array(
            cdd_matrix_input, inds_for_equality, RepType.INEQUALITY
        )
        # remove redundancies
        # cdd.matrix_canonicalize(cdd_matrix)
        joint_poly = cdd.polyhedron_from_matrix(cdd_matrix)
        # generators are vertices of the Polytope
        # this is in the form [t V] where t indicates whether
        # the vertex is part of a convex combination (t=1) or
        # t=0 if the vertex is part of a conic combination.
        # each row is a vertex
        joint_vertices_obj = cdd.copy_generators(joint_poly)
        if self.compute_exact_cond_dist:
            joint_vertices = [
                [mpq(str(v_ele)) for v_ele in vertex]
                for vertex in joint_vertices_obj.array
            ]
            joint_vertices = np.array(joint_vertices)
        else:
            joint_vertices = np.array(joint_vertices_obj.array).astype(np.float64)
        n_vertices = joint_vertices.shape[0]
        self.bookkeeping_add_intermed_vertex_count(n_vertices, file_enum, polytope)

        # check that each t value is 1.  By construction, the first
        # column will only contain 0 or 1s
        if sum(joint_vertices[:, 0]) < n_vertices:
            raise ValueError(
                "Polytope representing rule accs, class freq "
                + "constraints must be writeable as a convex combination. "
                + "cdd's vertex representation leads us to conclude the "
                + "polyhedron requires a conic combination to write"
            )
        if polytope == Polytope.POSSIBLE_PATTERNS_VREP:
            if self.compute_exact_cond_dist:
                transformed_vertices = joint_vertices[
                    :, 1:
                ] @ self.pattern_constraints_dense.T.astype(mpq)
            else:
                transformed_vertices = (
                    joint_vertices[:, 1:] @ self.pattern_constraints.T
                )

        elif polytope == Polytope.POSSIBLE_PATTERNS_VREP_REPARAM:
            inds_to_keep = np.arange(
                1, self.n_patterns * (self.n_classes + 1), self.n_classes + 1
            )

            transformed_vertices = joint_vertices[:, inds_to_keep]

        if not self.compute_exact_cond_dist:

            def cast_to_mpq(x):
                return mpq(str(x))

            transformed_vertices = np.array(
                np.vectorize(cast_to_mpq)(transformed_vertices)
            )

        transformed_cdd_input = list(
            np.hstack([np.ones((n_vertices, 1)).astype(int), transformed_vertices])
        )

        # new create another polyhedron based on the transformed vertices
        transformed_cdd_matrix = cdd.matrix_from_array(
            transformed_cdd_input, rep_type=RepType.GENERATOR
        )
        pattern_poly = cdd.polyhedron_from_matrix(transformed_cdd_matrix)
        pattern_hrep = cdd.copy_inequalities(pattern_poly)
        # if we're not computing exact quantities, add in the distribution
        # constraint because numerical errors could cause that
        # constraint to not appear in the resulting h-rep, which would
        # throw off the volume computation
        if not self.compute_exact_cond_dist:
            curr_array = list(pattern_hrep.array)
            lin_set = list(pattern_hrep.lin_set)
            lin_set.append(len(curr_array))
            curr_array.append(
                [float(self.datapoints)] + (-1 * np.ones(self.n_patterns)).tolist()
            )
            # make a new cdd.Matrix object since I don't know if changing parameters
            # this way will break cdd stuff
            print(curr_array)
            # print(lin_set)
            new_cdd_matrix = cdd.matrix_from_array(
                curr_array, lin_set=lin_set, rep_type=RepType.INEQUALITY
            )
            new_poly = cdd.polyhedron_from_matrix(new_cdd_matrix)
            pattern_hrep = cdd.copy_inequalities(new_poly)
        polytope_dim = self.n_patterns - len(pattern_hrep.lin_set)
        if not self.check_cdd_output_is_integral(str(pattern_hrep)):
            pattern_hrep = self.convert_cdd_output_to_ints(str(pattern_hrep))
        return str(pattern_hrep), polytope_dim

    def regenerate_hrep_file(self, file_enum: int, polytope: Polytope):
        # hold all the current info in temporary variables
        current_vals = {
            "true_accs": self.true_accs,
            "true_class_freqs": self.true_class_freqs,
        }

        # set the object vars to what we want them to be
        self.true_accs = self.bookkeeping_dict["true_accs_" + str(file_enum)].squeeze()
        self.true_class_freqs = self.bookkeeping_dict[
            "true_class_freqs_" + str(file_enum)
        ].squeeze()

        print(
            "Regenerating count input file from file_enum",
            file_enum,
            ", polytope ",
            polytope.name,
            " :",
            polytope.value,
        )
        if polytope == Polytope.EPS_OPT_PATTERNS:
            self.refresh_weights()

        self.construct_input_helper(file_enum, polytope)

        # put the original var values back and refresh weights
        self.true_accs = current_vals["true_accs"]
        self.true_class_freqs = current_vals["true_class_freqs"]
        self.refresh_weights()

    # run count in latte and save inputs
    def run_count_or_integrate(
        self,
        command: str,
        file_enum,
        polytope: Polytope,
        extra_args: List[str],
        # when rerunning a specific count command, override the results and
        # ignore the enforced failure we have put in
        is_rerun=False,
    ):
        # construct the arguments used
        input_fname = self.construct_hrep_input_fname(file_enum, polytope)
        if command == "integrate":
            cli_command = [command, "--valuation=volume", "--triangulate"]
        else:
            cli_command = [command]
        command_and_args = cli_command + extra_args
        command_and_args += [input_fname]
        print("--------------------------------------")
        print("Command: ", command)
        print("Polytope type: ", polytope.name)
        if self.compute_exact_cond_dist:
            print("Exact Result: ", True)
        else:
            print("Exact Result: False (Result is an Estimate)")
        print("Running: ", " ".join(command_and_args))
        # for some reason count alternates between writing to stdout and stderr
        # so we combine them into stdout
        comp_process = subprocess.run(
            command_and_args,
            # check is False by default, we want to record the results even
            # the polytope is empty, which gives return code != 0
            # check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.result = self.extract_result(command, comp_process.stdout)
        if command == "integrate":
            output_type = "Volume"
        else:
            output_type = "Counts"
        print("\nOutput " + output_type + ": ", self.result)

        # count spits out files in the current directory, so we'll move them
        # to the results directory.  We won't check for folder existence to
        # safeguard existing data.
        # integrate does not have these outputs though
        output_fnames = []
        if command == "count":
            output_fnames += [
                "numOfLatticePoints",
                "latte_stats",
                "totalTime",
            ]
            if "--maxdet=1000" not in POLYTOPE_COUNT_EXTRA_ARGS[polytope]:
                output_fnames += ["numOfUnimodularCones"]
        # hand files from checking if input polytope is empty
        # if using latte style input files
        if POLYTOPE_INPUT_TYPES[polytope] == "latte":
            output_fnames = [
                "Check_emp.lp",
                "Check_emp.lps",
            ] + output_fnames

            # this file has no useful information
            subprocess.run(["rm", "Check_emp.out"])

        if self.delete_outputs:
            for fname in output_fnames:
                subprocess.run(["rm", fname])
        else:
            self.log_dir = join(self.dir, command + "_run_" + str(file_enum))
            # don't make dir if it's a rerun
            # also don't make dir if it already exists and we're not on a rerun
            # in case you want to run both count and integrate on the same inputs
            if not os.path.isdir(self.log_dir) and not is_rerun:
                os.makedirs(self.log_dir)

            subdir = command + "_" + POLYTOPE_FILE_NAMES[polytope]
            if not is_rerun:
                os.makedirs(join(self.log_dir, subdir))

            output_dir = join(self.log_dir, subdir)
            for i, fname in enumerate(output_fnames):
                # skip trying to move non-existent files if count fails due to
                # empty polytope
                if command == "count" and comp_process.returncode != 0 and i == 3:
                    break
                subprocess.run(["mv", fname, join(output_dir, fname)])

        return comp_process.args, comp_process.stdout, comp_process.stderr

    def extract_result(self, command: str, stdout: str) -> mpq:
        if command == "count":
            fname = "numOfLatticePoints"

            try:
                with open(fname, "r") as f:
                    result = mpq(f.readline())
                    return mpq(result)
            except FileNotFoundError:
                # we must continue so logs from LattE call can be written!
                print(
                    "count extraction failed due to FileNotFoundError for numOfLatticePoints!"
                )
        elif command == "integrate":
            # here we need to read from stdout because there's not output files with the data
            lines = reversed(str.splitlines(stdout))
            for line in lines:
                if "Answer: " in line:
                    tokens = line.split(" ")
                    return mpq(tokens[-1])
        else:
            raise ValueError(
                "Can only extract results for 'count' or 'integrate' commands"
            )

        return mpq(-1)

    def write_logs(
        self, stdout: str, stderr: str, file_enum: int, polytope: Polytope, command: str
    ):
        fname_prefix = POLYTOPE_FILE_NAMES[polytope]
        stdout_fname = fname_prefix + "_" + command + "_" + str(file_enum) + ".log"
        stderr_fname = fname_prefix + "_" + command + "_err_" + str(file_enum) + ".log"
        if self.delete_outputs:
            stdout_fname = join(self.dir, stdout_fname)
            stderr_fname = join(self.dir, stderr_fname)
        else:
            if command not in ["count", "integrate"]:
                raise ValueError(
                    "Command must be 'count' or 'integrate' in `write_logs`"
                )
            stdout_fname = join(self.log_dir, stdout_fname)
            stderr_fname = join(self.log_dir, stderr_fname)

        def write_log_helper(contents: str, fname: str):
            fname_alt = fname + "_alternate"
            if os.path.exists(join(self.dir, fname)):
                # append here just in case there are multiple error
                with open(fname_alt, "a") as f:
                    f.write(contents)
                raise ValueError(
                    "File name ",
                    fname,
                    " already exists, writing contents to ",
                    fname_alt,
                )
            else:
                with open(fname, "w") as f:
                    f.write(contents)

        write_log_helper(stdout, stdout_fname)
        self.stdout_fname = stdout_fname

        if stderr is not None:
            write_log_helper(stderr, stderr_fname)
            self.stderr_fname = stderr_fname

    def count_and_integrate_patterns_helper(
        self,
        file_enum: int,
        polytope: Polytope,
        command: str,
        delete_hrep_file=False,
        is_rerun=False,
    ):
        if command not in ["count", "integrate"]:
            raise ValueError("command ", command, " is not supported")

        extra_args = copy.deepcopy(DEFAULT_EXTRA_ARGS)

        if command == "count":
            extra_args += POLYTOPE_COUNT_EXTRA_ARGS[polytope]

        if POLYTOPE_INPUT_TYPES[polytope] == "cdd":
            extra_args += ["--cdd"]  # these run types have cdd style input

        cli_command, stdout, stderr = self.run_count_or_integrate(
            command,
            file_enum,
            polytope,
            extra_args=extra_args,
            is_rerun=is_rerun,
        )
        self.write_logs(stdout, stderr, file_enum, polytope, command)
        self.bookkeeping_add_result_and_command(
            file_enum, polytope, self.result, cli_command
        )

        if polytope == Polytope.EPS_OPT_PATTERNS:
            self.bookkeeping_add_accs_class_freqs(file_enum)
        elif polytope in [
            Polytope.POSSIBLE_PATTERNS_VREP,
            Polytope.POSSIBLE_PATTERNS_VREP_REPARAM,
        ]:
            self.bookkeeping_add_ratio_result(self.result, file_enum, polytope, command)
        elif polytope == Polytope.POSSIBLE_PATTERNS_ALL:
            pass
        else:
            raise NotImplementedError(polytope.name, " is not implemented")

        self.bookkeeping_write()

        if delete_hrep_file:
            os.remove(self.construct_hrep_input_fname(file_enum, polytope))

    def count_or_integrate_patterns(
        self,
        command: str,
        file_enum,
        override_polytope=None,
        delete_hrep_file=False,
    ):
        if command not in ["count", "integrate"]:
            raise ValueError("command arg provided must be 'count' or 'integrate'.")

        if override_polytope is None:
            for polytope in Polytope:
                if polytope in self.enums_to_skip or (
                    command == "count" and polytope == Polytope.POSSIBLE_PATTERNS_ALL
                ):
                    continue

                self.count_and_integrate_patterns_helper(
                    file_enum,
                    polytope,
                    command,
                    delete_hrep_file=delete_hrep_file,
                )
        elif override_polytope is not None:
            if not self.bookkeeping_exists:
                raise ValueError(
                    "Cannot use overrides for counting without existing bookkeeping info"
                )
            self.count_and_integrate_patterns_helper(
                file_enum,
                override_polytope,
                command,
                delete_hrep_file=delete_hrep_file,
                is_rerun=True,
            )
        else:
            raise ValueError(
                "Cannot only supply only one of file/run overrides.  Must supply both or neither."
            )

    def construct_inputs(self):
        # advance enum if necessary if a run has failed so counting patterns
        # with fixed conditional distribution is always on an even number
        self.bookkeeping_update_enum()

        # decrement the enum since we incremented it to preserve failed run outputs
        # if we're using overrides, we don't want to change it
        file_enum = self.file_enum - 1
        print("Number of patterns/dimensions: ", self.n_patterns)
        polytopes = []
        for polytope in Polytope:
            if polytope in self.enums_to_skip:
                continue
            polytopes.append(polytope)
            print("Constructing input file for ", polytope.name)
            self.construct_input_helper(file_enum, polytope)
            self.bookkeeping_add_polytope_dims(file_enum, polytope)

        return file_enum, polytopes

    def bookkeeping_fname_prefix(self) -> str:
        specialists_kw = "specialist" if self.use_specialists else "generalist"
        exact_kw = "exact" if self.compute_exact_cond_dist else "estim"

        prefix = "_".join(
            [
                self.weighting_strategy.get_name_shorthand(),
                exact_kw,
                "p",
                str(self.n_rules),
                "k",
                str(self.n_classes),
                "dec",
                str(self.decimal_places),
                "rndeps",
                str(self.rounding_epsilon),
                "appeps",
                str(self.appropriateness_epsilon),
                specialists_kw,
                "n",
                str(self.datapoints),
            ]
        )
        prefix = prefix.replace("/", "_")
        return prefix

    def bookkeeping_init(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.bookkeeping_fname: str = (
            self.bookkeeping_fname_prefix() + "_bookkeeping.mat"
        )
        self.bookkeeping_fname = join(self.dir, self.bookkeeping_fname)

        proposed_bookkeeping_dict = {
            "weighting_strategy": self.weighting_strategy.get_name_shorthand(),
            "exact_computation": self.compute_exact_cond_dist,
            "n_rules": self.n_rules,
            "n_classes": self.n_classes,
            "decimal_places": self.decimal_places,
            "rounding_epsilon": self.rounding_epsilon,
            "appropriateness_epsilon": str(self.appropriateness_epsilon),
            "file_enum": 0,
        }
        if os.path.exists(self.bookkeeping_fname):
            print("Reading existing bookkeeping .mat file")
            self.bookkeeping_dict = loadmat(self.bookkeeping_fname)

            # dump keys with underscores so we don't get a warning message
            self.bookkeeping_dict.pop("__version__", None)
            self.bookkeeping_dict.pop("__header__", None)
            self.bookkeeping_dict.pop("__globals__", None)

            # since MATLAB stores stuff as 2d array, we need to recover the singular values
            self.bookkeeping_dict["weighting_strategy"] = np.ndarray.item(
                self.bookkeeping_dict["weighting_strategy"]
            )
            self.bookkeeping_dict["exact_computation"] = np.ndarray.item(
                self.bookkeeping_dict["exact_computation"]
            )
            self.bookkeeping_dict["n_rules"] = np.ndarray.item(
                self.bookkeeping_dict["n_rules"]
            )
            self.bookkeeping_dict["n_classes"] = np.ndarray.item(
                self.bookkeeping_dict["n_classes"]
            )
            self.bookkeeping_dict["rounding_epsilon"] = np.ndarray.item(
                self.bookkeeping_dict["rounding_epsilon"]
            )
            self.bookkeeping_dict["appropriateness_epsilon"] = np.ndarray.item(
                self.bookkeeping_dict["appropriateness_epsilon"]
            )
            self.bookkeeping_dict["decimal_places"] = np.ndarray.item(
                self.bookkeeping_dict["decimal_places"]
            )
            self.bookkeeping_dict["total_patterns"] = np.ndarray.item(
                self.bookkeeping_dict["total_patterns"]
            )
            self.bookkeeping_dict["scaled_simplex_vol"] = np.ndarray.item(
                self.bookkeeping_dict["scaled_simplex_vol"]
            )
            self.bookkeeping_dict["file_enum"] = np.ndarray.item(
                self.bookkeeping_dict["file_enum"]
            )
            # now do some checks to make sure the hyperparams are correct
            keys_to_check = [
                "weighting_strategy",
                "exact_computation",
                "n_rules",
                "n_classes",
                "decimal_places",
                "rounding_epsilon",
                "appropriateness_epsilon",
            ]
            for key in keys_to_check:
                dict_val = self.bookkeeping_dict.get(key)
                if dict_val is None:
                    raise ValueError(
                        "Existing bookkeeping dict missing value for key ", key
                    )
                expected_val = proposed_bookkeeping_dict[key]
                if dict_val != expected_val:
                    raise ValueError(
                        "Existing bookkeeping dict value",
                        dict_val,
                        "for key ",
                        key,
                        "does not match current hyperparam value ",
                        expected_val,
                    )
            self.bookkeeping_exists = True
        else:
            self.bookkeeping_dict = proposed_bookkeeping_dict
            self.bookkeeping_dict["total_patterns"] = str(
                stars_and_bars(self.datapoints, self.n_patterns)
            )
            self.bookkeeping_dict["scaled_simplex_vol"] = str(
                scaled_simplex_vol(self.datapoints, self.n_patterns)
            )
            self.bookkeeping_write()
            print(
                "No bookkeeping file found, writing new bookkeeping .mat file to ",
                self.bookkeeping_fname,
            )

        self.file_enum: int = self.bookkeeping_dict["file_enum"]

    def bookkeeping_write(self):
        savemat(self.bookkeeping_fname, self.bookkeeping_dict)

    def bookkeeping_update_enum(self):
        self.file_enum += 1
        self.bookkeeping_dict["file_enum"] = self.file_enum
        self.bookkeeping_write()

    def bookkeeping_add_result_and_command(
        self, file_enum: int, polytope: Polytope, count_or_vol: mpq, command: str
    ):
        if command[0] == "integrate":
            raw_type = "volume"
        else:
            raw_type = "count"

        new_counts_and_command = {
            POLYTOPE_FILE_NAMES[polytope]
            + "_"
            + raw_type
            + "_"
            + str(file_enum): np.array([str(count_or_vol)]),
            POLYTOPE_FILE_NAMES[polytope] + "_command_" + str(file_enum): np.array(
                [command]
            ),
        }
        self.bookkeeping_dict.update(new_counts_and_command)

    def bookkeeping_add_accs_class_freqs(self, file_enum: int):
        # cast to string elements for writing
        true_accs_str = [str(true_acc) for true_acc in self.true_accs]
        true_class_freqs_str = [
            str(true_class_freqs) for true_class_freqs in self.true_class_freqs
        ]
        accs_class_freqs = {
            "true_accs_" + str(file_enum): [true_accs_str],
            "true_class_freqs_" + str(file_enum): [true_class_freqs_str],
        }
        self.bookkeeping_dict.update(accs_class_freqs)
        print("True rule accuracies are: ", true_accs_str)
        print("True class frequencies are: ", true_class_freqs_str)

    def bookkeeping_add_ratio_result(
        self, count_or_vol: mpq, file_enum: int, polytope: Polytope, command: str
    ):
        if command == "count":
            raw_type = "count"
            prob_type = "mass"
            all_denom = mpq(self.bookkeeping_dict["total_patterns"])
        elif command == "integrate":
            raw_type = "volume"
            prob_type = "density"
            key_computed = (
                POLYTOPE_FILE_NAMES[Polytope.POSSIBLE_PATTERNS_ALL]
                + "_volume_"
                + str(file_enum)
            )
            use_computed_key = key_computed in self.bookkeeping_dict.keys()
            key = key_computed if use_computed_key else "scaled_simplex_vol"
            den_vol = self.bookkeeping_dict[key]
            if use_computed_key:
                den_vol = np.ndarray.item(den_vol)
            all_denom = mpq(den_vol)
        else:
            raise ValueError(
                "command for `bookkeeping_add_ratio_result` must be 'count' or 'integrate'"
            )
        polytope_suffix = POLYTOPE_FILE_NAMES[polytope]
        num = mpq(
            np.ndarray.item(
                self.bookkeeping_dict[
                    POLYTOPE_FILE_NAMES[Polytope.EPS_OPT_PATTERNS]
                    + "_"
                    + raw_type
                    + "_"
                    + str(file_enum)
                ]
            )
        )
        optimal_to_possible_ratio = gmpy2.div(num, count_or_vol)
        self.bookkeeping_dict[
            "opt_to_possible_"
            + prob_type
            + "_ratio_"
            + polytope_suffix
            + "_"
            + str(file_enum)
        ] = [str(optimal_to_possible_ratio)]

        possible_to_all_ratio = gmpy2.div(count_or_vol, all_denom)
        self.bookkeeping_dict[
            "possible_to_all_"
            + prob_type
            + "_ratio_"
            + polytope_suffix
            + "_"
            + str(file_enum)
        ] = [str(possible_to_all_ratio)]

        print("Optimal to possible patterns ratio: ", optimal_to_possible_ratio)
        print("Possible to all patterns ratio: ", possible_to_all_ratio)
        print("--------------------------------------")

    def bookkeeping_add_intermed_vertex_count(
        self, counts: int, file_enum: int, polytope: Polytope
    ):
        counts_type_suffix = POLYTOPE_FILE_NAMES[polytope]
        self.bookkeeping_dict[
            "intermed_vertex_count_" + counts_type_suffix + "_" + str(file_enum)
        ] = [str(counts)]
        print("Intermediate vertices count: ", counts)

    def bookkeeping_add_polytope_dims(self, file_enum: int, polytope: Polytope):
        dim = self.polytope_dims[polytope]
        name = POLYTOPE_FILE_NAMES[polytope]
        self.bookkeeping_dict["polytope_dimension_" + name + "_" + str(file_enum)] = [
            str(dim)
        ]
        print("Dimension of polytope is ", dim)
