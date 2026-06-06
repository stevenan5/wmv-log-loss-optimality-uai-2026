import cvxpy as cp
import numpy as np
from typing import List

from bf import BFConditional, BFJoint
from bf_solver_base import BFSolverBase, BFSolverJointAddOn


class BFPrimalSolverBase(BFSolverBase):
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
        solver="MOSEK",
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
        self.solver = solver
        self.labeling: cp.Variable = cp.Variable(
            self.n_classes * self.n_pts_or_patterns
        )
        self.constraints: List[cp.Constraint] = []
        return

    def construct_double_sided_constraints(
        self, var_exp, expected_vals, error_bars, scaling=1
    ):
        ev_shape = expected_vals.shape
        lb = np.maximum(expected_vals - error_bars[:, 0], np.zeros(ev_shape))
        ub = np.minimum(expected_vals + error_bars[:, 1], np.ones(ev_shape))
        return [var_exp >= scaling * lb, var_exp <= scaling * ub]

    def construct_rule_acc_constraints(self, labeling, scaling=1):
        induced_rule_accs = self.rule_pred_constraints @ labeling

        if self.acc_const_equality:
            return [induced_rule_accs == self.est_accs * scaling]

        return self.construct_double_sided_constraints(
            induced_rule_accs, self.est_accs, self.accs_error_bars, scaling
        )

    def construct_class_freq_constraints(self, labeling, scaling=1):
        induced_class_freqs = self.class_freq_constraints @ labeling

        if self.class_freq_const_equality:
            return [induced_class_freqs == self.est_class_freqs * scaling]

        return self.construct_double_sided_constraints(
            induced_class_freqs,
            self.est_class_freqs,
            self.class_freqs_error_bars,
            scaling,
        )

    def construct_distribution_constraints(self, labeling, is_joint=False):
        constraints = [labeling >= 0]
        if is_joint:
            return constraints + [cp.sum(labeling) == 1]
        else:
            reshaped_labeling = cp.reshape(
                labeling, (self.n_pts_or_patterns, self.n_classes), order="C"
            )
            return constraints + [
                cp.sum(reshaped_labeling, axis=1) == np.ones(self.n_pts_or_patterns)
            ]

    def construct_objective(self):
        return cp.Maximize(cp.entr(self.labeling))


class BFPrimalConditional(BFPrimalSolverBase):
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
        BFPrimalSolverBase.__init__(
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
        BFPrimalSolverBase.check_inputs(self)
        self.constraints = self.construct_constraints(self.labeling)
        return

    def construct_constraints(self, labeling):
        rule_acc_constraints = self.construct_rule_acc_constraints(
            labeling, scaling=self.bf.preds_per_rule
        )
        class_freq_constraints = self.construct_class_freq_constraints(
            labeling, scaling=self.n_pts_or_patterns
        )
        distribution_constraints = self.construct_distribution_constraints(labeling)
        constraints = (
            rule_acc_constraints + class_freq_constraints + distribution_constraints
        )
        return constraints

    def solve(self):
        obj = self.construct_objective()
        constraints = self.construct_constraints(self.labeling)
        self.construct_cvxpy_problem(obj, constraints)

        self.problem.solve(verbose=self.verbose)
        self.bf_prediction = self.labeling.value


class BFPrimalJoint(BFPrimalSolverBase, BFSolverJointAddOn):
    def __init__(
        self,
        bf_joint: BFJoint,
        est_accs,
        est_class_freqs,
        est_pattern_dist,
        accs_error_bars,
        class_freqs_error_bars,
        pattern_dist_error_bars,
        verbose=False,
    ):
        self.bf = bf_joint
        BFPrimalSolverBase.__init__(
            self,
            self.bf.n_rules,
            self.bf.n_classes,
            est_accs,
            est_class_freqs,
            accs_error_bars,
            class_freqs_error_bars,
            self.bf.rule_pred_constraints,
            "joint",
            verbose=verbose,
        )
        self.est_pattern_dist = est_pattern_dist

        BFPrimalSolverBase.check_inputs(self)
        BFSolverJointAddOn.__init__(self, est_pattern_dist, pattern_dist_error_bars)
        self.tau = self.n_pts_or_patterns
        self.constraints = self.construct_constraints(self.labeling)
        return

    def construct_pattern_constraints(self, labeling):
        induced_pattern_dist = self.bf.pattern_constraints @ labeling

        if self.fixed_pattern:
            return [induced_pattern_dist == self.est_pattern_dist]

        return self.construct_double_sided_constraints(
            induced_pattern_dist, self.est_pattern_dist, self.pattern_dist_error_bars
        )

    def construct_constraints(self, labeling):
        rule_acc_constraints = self.construct_rule_acc_constraints(labeling)
        class_freq_constraints = self.construct_class_freq_constraints(labeling)
        pattern_constraints = self.construct_pattern_constraints(labeling)
        distribution_constraints = self.construct_distribution_constraints(
            labeling, is_joint=True
        )
        constraints = (
            rule_acc_constraints
            + class_freq_constraints
            + pattern_constraints
            + distribution_constraints
        )
        return constraints

    def solve(self):
        obj = self.construct_objective()
        constraints = self.construct_constraints(self.labeling)
        self.construct_cvxpy_problem(obj, constraints)

        self.problem.solve(solver=self.solver, verbose=self.verbose)
        self.bf_prediction = self.labeling.value
