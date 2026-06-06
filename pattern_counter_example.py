import numpy as np
from gmpy2 import mpq
from weight_optimality import (
    WeightOptimality,
)
from weighting_strategy import OneCoinDawidSkeneWeights, MajorityVoteWeights

from weight_optimality import Polytope

if __name__ == "__main__":
    n_rules = 3
    n_classes = 2
    use_specialists = False
    # for hammer spammer
    # true_accs = np.array([mpq(9, 10), mpq(3, 5), mpq(3, 5)])
    # for all spammers
    true_accs = np.array([mpq(3, 5), mpq(3, 5), mpq(3, 5)])

    true_class_freqs = np.array([mpq(1, n_classes) for _ in range(n_classes)])

    ocds_weights = OneCoinDawidSkeneWeights(true_accs, true_class_freqs, n_classes)
    mv_weights = MajorityVoteWeights(true_accs, true_class_freqs)
    weighting_strategy = ocds_weights

    pattern_counter = WeightOptimality(
        weighting_strategy,
        n_rules,
        n_classes,
        true_accs,
        true_class_freqs,
        enums_to_skip=[
            Polytope.POSSIBLE_PATTERNS_ALL,
            # remove this if not using OCDS
            # Polytope.POSSIBLE_PATTERNS_VREP,
            # as the conversion to mpq is probably extremely slow.  This is
            # exactly computed when using OCDS
            Polytope.POSSIBLE_PATTERNS_VREP_REPARAM,
        ],
        use_specialists=use_specialists,
        delete_outputs=False,
        datapoints=1000,
        # this is the epsilon for `epsilon-optimal`
        appropriateness_epsilon=mpq(1, 10),  # can set to mpq(0,10) for exact result
        compute_exact_cond_dist=weighting_strategy.exp_weights_are_rational(),
    )

    file_enum, polytopes = pattern_counter.construct_inputs()

    pattern_counter.count_or_integrate_patterns("integrate", file_enum)
    # pattern_counter.count_or_integrate_patterns("count", file_enum)
