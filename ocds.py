import numpy as np
from bf import BFConditional
from mv import MV
from weighting_strategy import OneCoinDawidSkeneWeights


class OCDS(BFConditional):
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
        self.weight_creator = OneCoinDawidSkeneWeights(
            true_accs, true_class_freqs, n_classes
        )
        self.true_quantity_cond_dist = self.compute_true_quant_cond_dist()

    def e_step(self, accs, class_freqs):
        # normalize class frequencies just in case there is numerical error
        class_freqs /= np.sum(class_freqs)
        self.weight_creator.change_params(accs, class_freqs, None)
        self.em_rule_weights = self.weight_creator.construct_rule_weights()
        self.em_class_freqs_weights = (
            self.weight_creator.construct_class_freqs_weights()
        )
        return self.compute_prediction(
            self.em_rule_weights, self.em_class_freqs_weights, None
        )

    def m_step(self, labeling):
        return self.compute_induced_quantities_from_labeling(labeling)

    def em_alg(self, conv_eps=1e-3):
        # majority vote initialization
        mv = MV(
            self.n_rules,
            self.n_classes,
            self.true_accs,
            self.true_class_freqs,
            self.rule_pred_constraints,
        )
        putative_labeling = mv.compute_cond_dist()
        converged = False
        accs, class_freqs = self.m_step(putative_labeling)
        while not converged:
            prev_accs = accs
            prev_class_freqs = class_freqs
            putative_labeling = self.e_step(accs, class_freqs)
            accs, class_freqs = self.m_step(putative_labeling)

            # check for convergence
            if (
                np.linalg.norm(accs - prev_accs, ord=np.inf) < conv_eps
                and np.linalg.norm(class_freqs - prev_class_freqs, ord=np.inf)
                < conv_eps
            ):
                converged = True

        putative_labeling = self.e_step(accs, class_freqs)

        return putative_labeling

    def compute_true_quant_cond_dist(self):
        return self.e_step(self.true_accs, self.true_class_freqs)
