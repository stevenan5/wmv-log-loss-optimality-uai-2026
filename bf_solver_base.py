from abc import ABC, abstractmethod
from typing import Union, List

import cvxpy as cp
import numpy as np

from scipy.sparse import coo_array, csr_array

from make_constraints import (
    construct_class_freq_constraints,
)
from util import _check_shape


class BFSolverBase(ABC):
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
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.est_accs = est_accs
        self.est_class_freqs = est_class_freqs
        self.accs_error_bars = accs_error_bars
        self.class_freqs_error_bars = class_freqs_error_bars
        self.rule_pred_constraints = csr_array(coo_array(rule_pred_constraints))
        self.problem_type = problem_type
        self.verbose = verbose

        if self.problem_type not in ["joint", "conditional"]:
            raise ValueError("Problem type must be 'joint' or 'conditional'")

        self.check_acc_and_class_freq_equality()

        rpc_shape = rule_pred_constraints.shape
        if rpc_shape[0] != self.n_rules:
            raise ValueError(
                "Row count of rule_pred_constraints, ",
                rpc_shape[0],
                "must equal self.n_rules, ",
                self.n_rules,
            )
        if rpc_shape[1] % self.n_classes != 0:
            raise ValueError(
                "Column count of rule_pred_constraints must be divisible by self.n_classes: ",
                self.n_classes,
            )

        self.n_pts_or_patterns = rpc_shape[1] // self.n_classes
        self.class_freq_constraints = construct_class_freq_constraints(
            self.n_classes, self.n_pts_or_patterns
        )

    def check_inputs(self):
        items_to_check = [
            self.est_accs,
            self.est_class_freqs,
            self.accs_error_bars,
            self.class_freqs_error_bars,
        ]

        dims_of_items_to_check = [
            (self.n_rules,),
            (self.n_classes,),
            (self.n_rules, 2),
            (self.n_classes, 2),
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

    def check_acc_and_class_freq_equality(self):
        self.acc_const_equality = np.abs(self.accs_error_bars).sum() == 0
        self.class_freq_const_equality = np.abs(self.class_freqs_error_bars).sum() == 0

    def update_error_bars(self, accs_error_bars, class_freqs_error_bars):
        self.accs_error_bars = accs_error_bars
        self.class_freqs_error_bars = class_freqs_error_bars
        self.check_acc_and_class_freq_equality()

    @abstractmethod
    def construct_objective(self) -> Union[cp.Minimize, cp.Maximize]:
        pass

    @abstractmethod
    def construct_constraints(self, labeling, /) -> List[cp.Constraint]:
        pass

    def construct_cvxpy_problem(self, objective, constraints):
        objective = self.construct_objective()
        self.problem = cp.Problem(objective, constraints)

    @abstractmethod
    def solve(self):
        pass


class BFSolverJointAddOn:
    def __init__(
        self,
        est_pattern_dist,
        pattern_dist_error_bars,
    ):
        self.est_pattern_dist = est_pattern_dist
        self.pattern_dist_error_bars = pattern_dist_error_bars
        # if we have a fixed pattern, we won't instantiate the pattern
        # constraints and will directly implement a modified objective
        self.check_pattern_equality()

    def check_pattern_equality(self):
        self.fixed_pattern = np.abs(self.pattern_dist_error_bars).sum() == 0
