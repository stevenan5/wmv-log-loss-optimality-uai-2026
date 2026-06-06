from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.special import softmax

from make_constraints import (
    construct_class_freq_constraints,
    construct_pattern_constraints,
    construct_pred_constraints,
)
from util import _check_shape


class BFBase(ABC):
    def __init__(
        self,
        n_rules,
        n_classes,
        true_accs,
        true_class_freqs,
    ):
        self.n_rules = n_rules
        self.n_classes = n_classes
        self.true_accs = true_accs
        self.true_class_freqs = true_class_freqs

        # placeholders
        self.rule_pred_constraints = np.array([0])
        self.class_freq_constraints = np.array([0])
        BFBase.check_inputs(self)

    def check_inputs(self):
        items_to_check = [
            self.true_accs,
            self.true_class_freqs,
        ]

        dims_of_items_to_check = [
            (self.n_rules,),
            (self.n_classes,),
        ]

        for _, (item, dims) in enumerate(zip(items_to_check, dims_of_items_to_check)):
            _check_shape(item, dims)

    def check_rule_pred_constraints(self, rule_pred_constraints):
        rpc_shape = rule_pred_constraints.shape
        if rpc_shape[1] % self.n_classes != 0:
            raise ValueError(
                "Column count of custom pred constraint matrix must be divisible by ",
                self.n_classes,
                " (self.n_classes)",
            )
        if rpc_shape[0] != self.n_rules:
            raise ValueError(
                "Row count of custom pred constraint matrix must be ",
                self.n_rules,
                " (self.n_rules)",
            )

    # compute the accs that one would use to obtain the weights if they were to use
    # the OCDS weighting strategy
    def compute_accs_from_weights(self, accs_weights):
        _check_shape(accs_weights, (self.n_rules,))
        induced_accs = 1 / (1 + (self.n_classes - 1) * np.exp(-1 * accs_weights))
        return induced_accs

    def compute_class_freqs_from_weights(self, class_freqs_weights):
        _check_shape(class_freqs_weights, (self.n_classes,))
        induced_class_freqs = softmax(class_freqs_weights)
        return induced_class_freqs

    def construct_ds_cond_weights(self, param_accs, param_class_freqs):
        # assuming that param accs are between [0,1] and class freqs
        # are non-negative and sum to 1
        _check_shape(param_accs, (self.n_rules,))
        _check_shape(param_class_freqs, (self.n_rules,))

        rule_weights = np.log(param_accs * (self.n_classes - 1) / (1 - param_accs))
        class_freq_weights = np.log(param_class_freqs)

        return rule_weights, class_freq_weights

    def compute_conditional_scores(self, rule_weights, class_freq_weights):
        _check_shape(rule_weights, (self.n_rules,))
        _check_shape(class_freq_weights, (self.n_classes,))
        scores = (
            self.rule_pred_constraints.T @ rule_weights
            + self.class_freq_constraints.T @ class_freq_weights
        )
        return scores

    def conditional_softmax(self, scores):
        if scores.shape[0] % self.n_classes != 0:
            raise ValueError(
                "scores must have length which is a multiple of self.n_classes"
            )
        scores_reshaped = np.reshape(scores, (-1, self.n_classes))
        cond_softmax = softmax(scores_reshaped, axis=1)

        return cond_softmax.flatten()

    @abstractmethod
    def compute_induced_quantities(
        self, rule_weights, class_freq_weights, pattern_weights, /
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def compute_prediction(self, acc_weights, class_freq_weights, pattern_weights, /):
        pass


class BFConditional(BFBase):
    def __init__(
        self,
        n_rules,
        n_classes,
        true_accs,
        true_class_freqs,
        rule_pred_constraints,
    ):
        super().__init__(n_rules, n_classes, true_accs, true_class_freqs)
        self.rule_pred_constraints = rule_pred_constraints
        self.n_points = self.rule_pred_constraints.shape[1] // self.n_classes
        self.class_freq_constraints = construct_class_freq_constraints(
            self.n_classes, self.n_points
        )
        # compute the number of predictions per rule
        self.preds_per_rule = self.rule_pred_constraints.sum(axis=1)

        BFConditional.check_inputs(self)

    def check_inputs(self):
        self.check_rule_pred_constraints(self.rule_pred_constraints)

    def compute_induced_quantities(
        self,
        rule_weights,
        class_freq_weights,
        __pattern_weights__,
    ):
        scores = self.compute_conditional_scores(rule_weights, class_freq_weights)
        cond_pred = self.conditional_softmax(scores)

        # here, we return the accuracies of rules given that they predict,
        # i.e. Pr(h(X) = Y | h(X) != ?)
        ind_accs, ind_class_freqs = self.compute_induced_quantities_from_labeling(
            cond_pred
        )

        return ind_accs, ind_class_freqs, np.array([0])

    def compute_induced_quantities_from_labeling(self, labeling):
        ind_accs = self.rule_pred_constraints @ labeling / self.preds_per_rule
        ind_class_freqs = self.class_freq_constraints @ labeling / self.n_points
        return ind_accs, ind_class_freqs

    def compute_prediction(self, acc_weights, class_freq_weights, __pattern_weights__):
        scores = self.compute_conditional_scores(acc_weights, class_freq_weights)
        return self.conditional_softmax(scores)

    def set_accs_class_freqs(self, accs, class_freqs):
        self.true_accs = accs
        self.true_class_freqs = class_freqs


class BFJoint(BFBase):
    # if a custom rule_pred_constraints matrix is provided, we're expecting a subset of the possible patterns
    def __init__(
        self,
        n_rules,
        n_classes,
        true_accs,
        true_class_freqs,
        true_pattern_dist=None,
        use_specialists=False,
        rule_pred_constraints=None,
    ):
        super().__init__(n_rules, n_classes, true_accs, true_class_freqs)
        self.true_pattern_dist = true_pattern_dist
        self.use_specialists = use_specialists

        self.custom_pred_constraints = rule_pred_constraints is not None
        if rule_pred_constraints is not None:
            self.rule_pred_constraints = rule_pred_constraints
            self.n_patterns = self.rule_pred_constraints.shape[1] // self.n_classes
        else:
            self.rule_pred_constraints, self.patterns, self.n_patterns = (
                construct_pred_constraints(
                    self.n_rules, self.n_classes, self.use_specialists
                )
            )

        self.class_freq_constraints = construct_class_freq_constraints(
            self.n_classes, self.n_patterns
        )
        self.pattern_constraints = construct_pattern_constraints(
            self.n_classes, self.n_patterns
        )
        BFJoint.check_inputs(self)

    def check_inputs(self):
        if self.true_pattern_dist is not None:
            _check_shape(self.true_pattern_dist, (self.n_patterns,))

        if self.custom_pred_constraints:
            self.check_rule_pred_constraints(self.rule_pred_constraints)

    def compute_pattern_dist_from_weights(
        self,
        rule_weights,
        class_freq_weights,
        pattern_weights,
    ):
        _check_shape(pattern_weights, (self.n_patterns,))

        scores = self.compute_conditional_scores(rule_weights, class_freq_weights)
        pattern_sum = self.pattern_constraints @ np.exp(scores)
        induced_pattern_dist = np.exp(pattern_weights) * pattern_sum
        # ensure that the pattern distribution sums to 1
        induced_pattern_dist /= induced_pattern_dist.sum()

        return induced_pattern_dist

    def construct_pattern_weights(self, rule_weights, class_freq_weights, pattern_dist):
        _check_shape(pattern_dist, (self.n_patterns,))
        scores = self.compute_conditional_scores(rule_weights, class_freq_weights)
        pattern_weights = np.log(
            pattern_dist / (self.pattern_constraints @ np.exp(scores))
        )
        return pattern_weights

    def compute_induced_quantities(
        self, rule_weights, class_freqs_weights, pattern_weights
    ):
        scores = self.compute_joint_scores(
            rule_weights, class_freqs_weights, pattern_weights
        )
        joint_pred = softmax(scores)
        # here, we return the probability that a rule's prediction matches the label,
        # i.e. Pr(h(X) = Y)
        ind_accs = self.rule_pred_constraints @ joint_pred
        ind_class_freqs = self.class_freq_constraints @ joint_pred
        ind_pattern_dist = self.pattern_constraints @ joint_pred

        return ind_accs, ind_class_freqs, ind_pattern_dist

    def create_ds_joint_dist(
        self, param_accs, param_class_freqs, pattern_dist=None, faithful=False
    ):
        if faithful and self.use_specialists:
            raise ValueError(
                "Cannot construct faithful OCDS joint distribution when there are specialists"
            )
        if not faithful and pattern_dist is None:
            raise ValueError(
                "Must provide pattern_dist when computing unfaithful OCDS joint dist"
            )
        if pattern_dist is not None:
            _check_shape(pattern_dist, (self.n_patterns,))

        ds_rule_weights, ds_class_freq_weights = self.construct_ds_cond_weights(
            param_accs, param_class_freqs
        )
        # now enforce the pattern distribution
        if faithful:
            pattern_weights = 0
        else:
            pattern_weights = self.construct_pattern_weights(
                ds_rule_weights, ds_class_freq_weights, pattern_dist
            )

        scores = self.compute_joint_scores(
            ds_rule_weights, ds_class_freq_weights, pattern_weights
        )

        self.ds_joint_dist = softmax(scores)

    def compute_joint_scores(self, acc_weights, class_freqs_weights, pattern_weights):
        scores = self.compute_conditional_scores(acc_weights, class_freqs_weights)
        scores += self.pattern_constraints.T @ pattern_weights
        return scores

    def compute_prediction(self, acc_weights, class_freqs_weights, pattern_weights):
        scores = self.compute_joint_scores(
            acc_weights, class_freqs_weights, pattern_weights
        )
        return softmax(scores)
