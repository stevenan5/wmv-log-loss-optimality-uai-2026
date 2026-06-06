import numpy as np
from bf import BFConditional
from weighting_strategy import MajorityVoteWeights


class MV(BFConditional):
    def __init__(
        self,
        n_rules,
        n_classes,
        true_accs,
        true_class_freqs,
        rule_pred_constraints,
    ):
        super().__init__(
            n_rules,
            n_classes,
            true_accs,
            true_class_freqs,
            rule_pred_constraints,
        )
        self.weight_creator = MajorityVoteWeights(true_accs, true_class_freqs)

    def compute_cond_dist(self):
        return self.compute_prediction(
            np.ones(self.n_rules), np.zeros(self.n_classes), None
        )
