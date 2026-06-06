import cvxpy as cp
import numpy as np

from bf import BFConditional, BFJoint
from bf_solver_base import BFSolverBase, BFSolverJointAddOn
from util import _check_shape


class BFDualSolverBase(BFSolverBase):
    def __init__(
        self,
        n_rules,
        n_classes,
        est_accs,
        est_class_freqs,
        accs_error_bars,
        class_freqs_error_bars,
        rule_pred_constraints,
        problem_type,
        verbose=False,
    ):
        super().__init__(
            n_rules,
            n_classes,
            est_accs,
            est_class_freqs,
            accs_error_bars,
            class_freqs_error_bars,
            rule_pred_constraints,
            problem_type,
            verbose=verbose,
        )

        # rule or LF weights
        self.sigma = cp.Variable(self.n_rules)
        # class frequency weights
        self.gamma = cp.Variable(self.n_classes)

    def compute_error_bar_sizes(self, error_estimates, scaling=1):
        return scaling * np.maximum(
            error_estimates[:, 0],
            error_estimates[:, 1],
        )

    def construct_sigma_gamma_terms(self, sigma_scaling=1, gamma_scaling=1):
        if self.acc_const_equality:
            self.accs_eps = np.zeros(self.n_rules)
            sigma_abs_term = 0
        else:
            self.accs_eps = self.compute_error_bar_sizes(
                self.accs_error_bars, sigma_scaling
            )
            sigma_abs_term = cp.abs(self.sigma) @ self.accs_eps

        if self.class_freq_const_equality:
            self.class_freq_eps = 0
            gamma_abs_term = 0
        else:
            self.class_freq_eps = self.compute_error_bar_sizes(
                self.class_freqs_error_bars, gamma_scaling
            )
            gamma_abs_term = cp.abs(self.gamma) @ self.class_freq_eps

        sigma_term = self.sigma @ (sigma_scaling * self.est_accs)
        gamma_term = self.gamma @ (gamma_scaling * self.est_class_freqs)

        return sigma_term + gamma_term - sigma_abs_term - gamma_abs_term


class BFDualSolverConditional(BFDualSolverBase):
    def __init__(
        self,
        bf_conditional: BFConditional,
        est_accs,
        est_class_freqs,
        accs_error_bars,
        class_freqs_error_bars,
        verbose=False,
    ):
        self.bf = bf_conditional
        BFDualSolverBase.__init__(
            self,
            self.bf.n_rules,
            self.bf.n_classes,
            est_accs,
            est_class_freqs,
            accs_error_bars,
            class_freqs_error_bars,
            self.bf.rule_pred_constraints,
            "conditional",
            verbose=verbose,
        )
        BFDualSolverBase.check_inputs(self)
        self.n_points = self.n_pts_or_patterns

    def construct_objective(self):
        # scale these terms so the constraint matrix doesn't need scaling.
        # I.e. b/c we're working with conditional probabilities, we will
        # get sums of probabilities masses which could each be at most 1
        sigma_gamma_terms = self.construct_sigma_gamma_terms(
            self.bf.preds_per_rule, self.n_points
        )

        scores = self.bf.compute_conditional_scores(self.sigma, self.gamma)
        scores = cp.reshape(scores, (self.n_points, self.n_classes), order="C")
        lse = cp.sum(cp.log_sum_exp(scores, axis=1))

        obj = cp.Maximize(sigma_gamma_terms - lse)

        return obj

    def construct_constraints(self, __labeling__):
        return []

    def solve(self):
        obj = self.construct_objective()
        constraints = self.construct_constraints(None)
        self.construct_cvxpy_problem(obj, constraints)

        self.problem.solve(verbose=self.verbose)

        self.acc_weights = self.sigma.value
        self.class_freqs_weights = self.gamma.value
        self.bf_prediction = self.bf.compute_prediction(
            self.acc_weights, self.class_freqs_weights, None
        )


class BFDualSolverJoint(BFDualSolverBase, BFSolverJointAddOn):
    def __init__(
        self,
        bf_joint: BFJoint,
        est_accs,
        est_class_freqs,
        est_pattern_dist,
        accs_error_bars,
        class_freqs_error_bars,
        pattern_dist_error_bars,
        rule_pred_constraints,
        verbose=False,
    ):
        self.bf = bf_joint
        BFDualSolverBase.__init__(
            self,
            self.bf.n_rules,
            self.bf.n_classes,
            est_accs,
            est_class_freqs,
            accs_error_bars,
            class_freqs_error_bars,
            rule_pred_constraints,
            "joint",
            verbose,
        )

        BFSolverJointAddOn.__init__(self, est_pattern_dist, pattern_dist_error_bars)

        BFDualSolverBase.check_inputs(self)
        BFDualSolverJoint.check_inputs(self)

        self.tau = self.n_pts_or_patterns
        if not self.fixed_pattern:
            self.nu = cp.Variable(self.tau)

    def check_inputs(self):
        items_to_check = [
            self.est_pattern_dist,
            self.pattern_dist_error_bars,
        ]

        dims_of_items_to_check = [
            (self.tau,),
            (self.tau, 2),
        ]

        for i, (item, dims) in enumerate(zip(items_to_check, dims_of_items_to_check)):
            try:
                _check_shape(item, dims)
            except ValueError:
                raise ValueError(
                    "Element ",
                    i,
                    " of items_to_check has dimension mismatch.  Expected dims ",
                    dims,
                    "but got ",
                    item.shape,
                )

    def construct_objective(self):
        sigma_gamma_terms = self.construct_sigma_gamma_terms()

        if self.fixed_pattern:
            xi_terms = 0
            cond_scores = self.bf.compute_conditional_scores(self.sigma, self.gamma)
            lse = cp.sum(cp.log_sum_exp(cond_scores, axis=1))
            translation = cp.entr(self.est_pattern_dist)
        else:
            translation = 0
            # construct terms for the pattern distribution
            pattern_eps = self.compute_error_bar_sizes(self.pattern_dist_error_bars)
            xi_term = self.nu @ self.est_pattern_dist
            xi_abs_term = cp.abs(self.nu) @ pattern_eps
            xi_terms = xi_term - xi_abs_term

            # construct the log-sum-exp term
            joint_scores = self.bf.compute_joint_scores(self.sigma, self.gamma, self.nu)
            lse = cp.log_sum_exp(joint_scores)

        obj = cp.Maximize(sigma_gamma_terms + xi_terms - lse - translation)

        return obj

    def construct_constraints(self, __labeling__):
        return []

    def solve(self):
        obj = self.construct_objective()
        constraints = self.construct_constraints(None)
        self.construct_cvxpy_problem(obj, constraints)

        self.problem.solve(verbose=self.verbose)

        self.acc_weights = self.sigma.value
        self.class_freqs_weights = self.gamma.value
        if self.fixed_pattern:
            self.pattern_weights = self.bf.construct_pattern_weights(
                self.sigma, self.gamma, self.est_pattern_dist
            )
        else:
            self.pattern_weights = self.nu.value

        self.bf_prediction = self.bf.compute_prediction(
            self.acc_weights, self.class_freqs_weights, self.pattern_weights
        )
