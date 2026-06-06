from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
import gmpy2

EPSILON = 1e-4


class WeightingStrategy(ABC):
    def __init__(self, rule_accs, class_freqs, pattern_dist, /):
        self.change_params(rule_accs, class_freqs, pattern_dist)

    def change_params(self, rule_accs, class_freqs, pattern_dist, /):
        self.rule_accs = rule_accs
        self.class_freqs = class_freqs
        self.pattern_dist = pattern_dist

        self.check_inputs()

    def check_inputs(self):
        self.check_bounds(self.rule_accs, 0, 1)
        self.check_bounds(self.class_freqs, 0, 1)
        self.check_sums_to_one(self.class_freqs)
        if self.pattern_dist is not None:
            self.check_bounds(self.pattern_dist, 0, 1)
            self.check_sums_to_one(self.pattern_dist)

    def check_bounds(self, quantities, lower_bound, upper_bound):
        if np.any(quantities < lower_bound) or np.any(quantities > upper_bound):
            raise ValueError(
                "A provided quantity does not satisfy the lower bound ",
                lower_bound,
                " and/or the upper bound ",
                upper_bound,
            )

    def check_sums_to_one(self, distribution):
        prob_masses = np.sum(distribution)
        if not np.isclose(prob_masses, 1):
            raise ValueError(
                "A provided distribution sums to ", prob_masses, "which is not 1."
            )

    def update_accs_class_freqs(self, accs, class_freqs):
        self.rule_accs = accs
        self.class_freqs = class_freqs

    @abstractmethod
    def construct_rule_weights(self) -> npt.NDArray:
        pass

    @abstractmethod
    def construct_class_freqs_weights(self) -> npt.NDArray:
        pass

    @abstractmethod
    def get_name_shorthand(self) -> str:
        pass

    # whether taking the exponential of the weights gives a rational number
    # when the rule accuracies and class frequencies are rational
    @abstractmethod
    def exp_weights_are_rational(self) -> bool:
        pass

    def get_exp_weights(self):
        raise NotImplementedError("Exponentiated weights are not rational")


class OneCoinDawidSkeneWeights(WeightingStrategy):
    def __init__(self, rule_accs, class_freqs, n_classes):
        self.n_classes = n_classes
        super().__init__(rule_accs, class_freqs, None)

    def construct_rule_weights(self):
        clipped_rule_accs = np.clip(self.rule_accs.astype(float), EPSILON, 1 - EPSILON)
        return np.log(clipped_rule_accs * (self.n_classes - 1)) - np.log(
            1 - clipped_rule_accs
        )

    def construct_class_freqs_weights(self):
        clipped_class_freqs = np.clip(
            self.class_freqs.astype(float), EPSILON, 1 - EPSILON
        )
        clipped_class_freqs /= clipped_class_freqs.sum()
        return np.log(clipped_class_freqs)

    def get_name_shorthand(self) -> str:
        return "ocds"

    def exp_weights_are_rational(self) -> bool:
        return True

    def get_exp_weights(self):
        accs = self.rule_accs.tolist()
        accs_mpq = [gmpy2.mpq(str(acc)) for acc in accs]
        class_freqs = self.class_freqs.tolist()
        class_freqs_weights_exp = [gmpy2.mpq(class_freq) for class_freq in class_freqs]
        rule_weights_exp = [
            gmpy2.div(gmpy2.mul(acc, self.n_classes - 1), 1 - acc) for acc in accs_mpq
        ]
        return np.array(rule_weights_exp + class_freqs_weights_exp)


class MajorityVoteWeights(WeightingStrategy):
    def __init__(self, rule_accs, class_freqs):
        super().__init__(rule_accs, class_freqs, None)

    def construct_rule_weights(self):
        return np.ones(self.rule_accs.shape)

    def construct_class_freqs_weights(self):
        return np.ones(self.class_freqs.shape)

    def get_name_shorthand(self) -> str:
        return "mv"

    def exp_weights_are_rational(self) -> bool:
        return False
